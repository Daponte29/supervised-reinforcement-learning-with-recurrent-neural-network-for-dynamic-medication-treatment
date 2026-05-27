variable "project_name" {
  description = "Short slug used to name all resources"
  type        = string
  default     = "my-project"   # TODO: rename
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "env" {
  description = "Deployment environment (dev | staging | prod)"
  type        = string
  default     = "dev"
}
