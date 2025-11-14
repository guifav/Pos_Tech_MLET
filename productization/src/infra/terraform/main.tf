# Core infrastructure for the LSTM productization platform
# Provisioning includes AKS, Cosmos DB, Azure Database for PostgreSQL, API Management,
# Azure Container Registry, and Blob Storage for MLflow artifacts.

resource "azurerm_resource_group" "main" {
  name     = local.rg_name
  location = var.location
  tags     = local.resource_tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = local.log_analytics_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.resource_tags
}

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${local.base_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.60.0.0/16"]
  tags                = local.resource_tags
}

resource "azurerm_subnet" "aks" {
  name                 = "snet-aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.60.0.0/20"]

  delegation {
    name = "aks-delegation"

    service_delegation {
      name = "Microsoft.ContainerService/managedClusters"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/action",
        "Microsoft.Network/virtualNetworks/subnets/join/action"
      ]
    }
  }

  service_endpoints = ["Microsoft.Storage", "Microsoft.ContainerRegistry"]
}

resource "azurerm_subnet" "postgres" {
  name                 = "snet-postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.60.32.0/24"]

  delegation {
    name = "postgres-delegation"

    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
        "Microsoft.Network/virtualNetworks/subnets/prepareNetworkPolicies/action"
      ]
    }
  }
}

resource "azurerm_container_registry" "main" {
  name                = local.acr_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Premium"
  admin_enabled       = false
  tags                = local.resource_tags
}

resource "azurerm_kubernetes_cluster" "main" {
  name                = local.aks_cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${local.base_name}-k8s"
  kubernetes_version  = var.aks_kubernetes_version
  tags                = local.resource_tags

  default_node_pool {
    name                = "system"
    vm_size             = var.aks_system_node_vm_size
    node_count          = var.aks_system_node_count
    type                = "VirtualMachineScaleSets"
    only_critical_addons_enabled = true
    vnet_subnet_id      = azurerm_subnet.aks.id
    node_labels = {
      "pool" = "system"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  linux_profile {
    admin_username = var.aks_admin_username

    ssh_key {
      key_data = var.aks_admin_ssh_public_key
    }
  }

  azure_active_directory_role_based_access_control {
    azure_rbac_enabled = true
    admin_group_object_ids = []
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
    dns_service_ip = "10.0.0.10"
    service_cidr   = "10.0.0.0/16"
    outbound_type  = "loadBalancer"
  }

  api_server_access_profile {
    authorized_ip_ranges = []
  }
}

# GPU node pools for training, updating, and serving
resource "azurerm_kubernetes_cluster_node_pool" "training" {
  name                  = local.aks_training_pool_name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.aks_gpu_node_vm_size
  node_count            = var.aks_gpu_node_count
  mode                  = "User"
  vnet_subnet_id        = azurerm_subnet.aks.id
  auto_scaling_enabled  = true
  min_count             = 0
  max_count             = max(2, var.aks_gpu_node_count)
  node_labels = {
    "workload" = "training"
    "accelerator" = "gpu"
  }
  node_taints = ["workload=training:NoSchedule"]
}

resource "azurerm_kubernetes_cluster_node_pool" "update" {
  name                  = local.aks_update_pool_name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.aks_gpu_node_vm_size
  node_count            = var.aks_gpu_node_count
  mode                  = "User"
  vnet_subnet_id        = azurerm_subnet.aks.id
  auto_scaling_enabled  = true
  min_count             = 0
  max_count             = max(2, var.aks_gpu_node_count)
  node_labels = {
    "workload" = "update"
    "accelerator" = "gpu"
  }
  node_taints = ["workload=update:NoSchedule"]
}

resource "azurerm_kubernetes_cluster_node_pool" "serving" {
  name                  = local.aks_serving_pool_name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.aks_gpu_node_vm_size
  node_count            = var.aks_gpu_node_count
  mode                  = "User"
  vnet_subnet_id        = azurerm_subnet.aks.id
  auto_scaling_enabled  = true
  min_count             = 0
  max_count             = max(2, var.aks_gpu_node_count)
  node_labels = {
    "workload" = "serving"
    "accelerator" = "gpu"
  }
  node_taints = ["workload=serving:NoSchedule"]
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
}

resource "azurerm_storage_account" "mlflow" {
  name                     = local.storage_account_name
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "ZRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"
  tags                     = local.resource_tags

  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_storage_container" "mlflow" {
  name                  = var.mlflow_artifact_container
  storage_account_id    = azurerm_storage_account.mlflow.id
  container_access_type = "private"
}

resource "azurerm_cosmosdb_account" "feedback" {
  name                = local.cosmos_account_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = var.cosmos_offer_type
  kind                = var.cosmos_account_kind
  ip_range_filter     = ""
  tags                = local.resource_tags

  consistency_policy {
    consistency_level       = "Session"
    max_interval_in_seconds = 5
    max_staleness_prefix    = 100
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }
}

resource "azurerm_cosmosdb_sql_database" "feedback" {
  name                = local.cosmos_sql_database
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.feedback.name
  throughput          = 400
}

resource "azurerm_cosmosdb_sql_container" "feedback" {
  name                = local.cosmos_sql_container
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.feedback.name
  database_name       = azurerm_cosmosdb_sql_database.feedback.name
  partition_key_paths = ["/predictionId"]
  throughput          = 400

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/\"_etag\"/?"
    }
  }
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                   = local.postgres_server_name
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = var.postgres_version
  administrator_login    = var.postgres_admin_username
  administrator_password = var.postgres_admin_password
  sku_name               = var.postgres_sku_name
  storage_mb             = var.postgres_storage_mb
  backup_retention_days        = 7
  geo_redundant_backup_enabled = false
  high_availability {
    mode = "ZoneRedundant"
  }
  maintenance_window {
    day_of_week  = 0
    start_hour   = 0
    start_minute = 0
  }
  public_network_access_enabled = true
  tags = local.resource_tags
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure" {
  name             = "allow-azure"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_postgresql_flexible_server_database" "financial_series" {
  name      = local.postgres_database_name
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.UTF8"
}

resource "azurerm_api_management" "main" {
  name                = local.apim_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  publisher_name      = var.apim_publisher_name
  publisher_email     = var.apim_publisher_email
  sku_name            = var.apim_sku_name
  tags                = local.resource_tags

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_api_management_api" "lstm_service" {
  name                = "lstm-service"
  resource_group_name = azurerm_resource_group.main.name
  api_management_name = azurerm_api_management.main.name
  revision            = "1"
  display_name        = "LSTM Model Service"
  path                = "lstm"
  protocols           = ["https"]
  service_url         = var.apim_backend_url
}

resource "azurerm_api_management_api_operation" "lstm_proxy" {
  operation_id        = "proxy"
  api_name            = azurerm_api_management_api.lstm_service.name
  api_management_name = azurerm_api_management.main.name
  resource_group_name = azurerm_resource_group.main.name
  display_name        = "Proxy to AKS"
  method              = "POST"
  url_template        = "/{path}"
  response {
    status_code = 200
    description = "Success"
  }
  request {
    query_parameter {
      name     = "path"
      required = false
      type     = "string"
    }
  }
}
