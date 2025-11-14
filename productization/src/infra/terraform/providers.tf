terraform {
  required_version = ">= 1.8.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.5"
    }
  }
}

provider "azurerm" {
  features {}

  # Use managed identity or environment-based authentication.
  use_msi         = true
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}
