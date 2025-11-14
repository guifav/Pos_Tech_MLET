output "resource_group_name" {
  description = "Name of the resource group hosting the infrastructure."
  value       = azurerm_resource_group.main.name
}

output "aks_name" {
  description = "Name of the AKS cluster."
  value       = azurerm_kubernetes_cluster.main.name
}

output "aks_kube_config" {
  description = "Raw kubeconfig for the AKS cluster."
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

output "container_registry_login_server" {
  description = "Login server URL of the Azure Container Registry."
  value       = azurerm_container_registry.main.login_server
}

output "cosmosdb_endpoint" {
  description = "Endpoint of the Cosmos DB account."
  value       = azurerm_cosmosdb_account.feedback.endpoint
}

output "postgres_fqdn" {
  description = "Fully qualified domain name for the PostgreSQL Flexible Server."
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "apim_gateway_url" {
  description = "Gateway URL for the API Management instance."
  value       = azurerm_api_management.main.gateway_url
}

output "mlflow_storage_account" {
  description = "Name of the storage account storing MLflow artifacts."
  value       = azurerm_storage_account.mlflow.name
}
