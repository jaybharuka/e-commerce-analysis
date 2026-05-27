# ============================================================
# Input Variables
# ============================================================

variable "aws_region" {
  description = "AWS region to deploy resources in."
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 instance type. t3.micro used here as t2.micro is not free-tier eligible on all accounts."
  type        = string
  default     = "t3.micro"
}

variable "app_port" {
  description = "Port the Streamlit application listens on."
  type        = number
  default     = 8501
}

variable "ssh_port" {
  description = "SSH port for management access."
  type        = number
  default     = 22
}

variable "key_name" {
  description = "Name of the EC2 Key Pair for SSH access. Create one in AWS Console > EC2 > Key Pairs, then download the .pem file. Set to null to skip SSH key attachment."
  type        = string
  default     = null
}

variable "dockerhub_image" {
  description = "Full DockerHub image reference to pull and run on the EC2 instance. Format: username/repo:tag"
  type        = string
  # Passed at runtime from the Jenkinsfile via -var flag.
  # Example: jaybharuka/ecommerce-analytics:42
}

variable "project_name" {
  description = "Logical name used as a prefix for all AWS resources."
  type        = string
  default     = "ecommerce-analytics"
}

variable "environment" {
  description = "Deployment environment label (production | staging | dev)."
  type        = string
  default     = "production"
}

variable "grafana_admin_password" {
  description = "Grafana admin password. Pass via TF_VAR_grafana_admin_password env var or Jenkins secret; never hardcode."
  type        = string
  sensitive   = true
  # No default — must be supplied explicitly so the pipeline fails fast
  # if the secret is not configured rather than silently using a weak value.
}
