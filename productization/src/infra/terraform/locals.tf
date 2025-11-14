locals {
  base_name    = lower(replace("${var.prefix}-${var.environment}", " ", ""))
  resource_tags = merge(
    {
      environment = var.environment
      owner       = "ml-ops"
      workload    = "lstm-productization"
    },
    var.tags
  )

  rg_name                 = "rg-${local.base_name}"
  log_analytics_name      = "law-${local.base_name}"
  aks_cluster_name        = "aks-${local.base_name}"
  aks_training_pool_name  = "train"
  aks_serving_pool_name   = "serve"
  aks_update_pool_name    = "update"
  acr_name                = substr(replace("acr${local.base_name}", "-", ""), 0, 50)
  storage_account_name    = substr(replace("st${local.base_name}", "-", ""), 0, 24)
  cosmos_account_name     = substr(replace("cosmos${local.base_name}", "-", ""), 0, 44)
  cosmos_sql_database     = "feedback-db"
  cosmos_sql_container    = "feedback"
  postgres_server_name    = "pg-${local.base_name}"
  postgres_database_name  = "financial_series"
  apim_name               = "apim-${local.base_name}"
}
