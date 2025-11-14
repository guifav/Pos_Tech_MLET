variable "subscription_id" {
  type        = string
  description = "Azure subscription ID where resources will be provisioned."
}

variable "tenant_id" {
  type        = string
  description = "Azure AD tenant ID used for authentication."
}

variable "prefix" {
  type        = string
  description = "Prefix applied to all resource names."
}

variable "environment" {
  type        = string
  description = "Deployment environment (e.g., dev, qa, prod)."
}

variable "location" {
  type        = string
  description = "Azure region for resource deployment."
  default     = "eastus"
}

variable "tags" {
  description = "Additional tags to apply to all resources."
  type        = map(string)
  default     = {}
}

variable "aks_kubernetes_version" {
  type        = string
  description = "AKS Kubernetes version."
  default     = "1.30.3"
}

variable "aks_system_node_vm_size" {
  type        = string
  description = "VM size for AKS system node pool."
  default     = "Standard_D4s_v5"
}

variable "aks_admin_username" {
  type        = string
  description = "Admin username for AKS nodes."
  default     = "azureuser"
}

variable "aks_admin_ssh_public_key" {
  type        = string
  description = "SSH public key for the AKS admin user."
}

variable "aks_gpu_node_vm_size" {
  type        = string
  description = "VM size for AKS GPU node pools used for training and inference."
  default     = "Standard_NC6s_v3"
}

variable "aks_system_node_count" {
  type        = number
  description = "Node count for the AKS system node pool."
  default     = 2
}

variable "aks_gpu_node_count" {
  type        = number
  description = "Node count per GPU node pool."
  default     = 1
}

variable "postgres_admin_username" {
  type        = string
  description = "Administrator username for Azure Database for PostgreSQL Flexible Server."
  default     = "pgadmin"
}

variable "postgres_admin_password" {
  type        = string
  description = "Administrator password for Azure Database for PostgreSQL Flexible Server."
  sensitive   = true
}

variable "postgres_version" {
  type        = string
  description = "PostgreSQL server version."
  default     = "16"
}

variable "postgres_sku_name" {
  type        = string
  description = "SKU for PostgreSQL Flexible Server."
  default     = "GP_Standard_D4s_v3"
}

variable "postgres_storage_mb" {
  type        = number
  description = "Storage size in MB for PostgreSQL."
  default     = 262144
}

variable "cosmos_account_kind" {
  type        = string
  description = "Cosmos DB account kind."
  default     = "GlobalDocumentDB"
}

variable "cosmos_offer_type" {
  type        = string
  description = "Cosmos DB offer type."
  default     = "Standard"
}

variable "apim_publisher_name" {
  type        = string
  description = "Publisher name for API Management."
}

variable "apim_publisher_email" {
  type        = string
  description = "Publisher email for API Management."
}

variable "apim_sku_name" {
  type        = string
  description = "SKU name for API Management instance."
  default     = "Developer_1"
}

variable "apim_backend_url" {
  type        = string
  description = "Backend service URL exposed by AKS ingress to be fronted by API Management."
}

variable "mlflow_artifact_container" {
  type        = string
  description = "Name of the Blob Storage container for MLflow artifacts."
  default     = "mlflow-artifacts"
}
