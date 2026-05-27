terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # TODO: configure remote state backend
  # backend "s3" {
  #   bucket = "my-tf-state"
  #   key    = "my_project/terraform.tfstate"
  #   region = var.aws_region
  # }
}

provider "aws" {
  region = var.aws_region
}

# ── S3 data bucket ────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.env}"
  # TODO: add lifecycle rules, versioning, encryption
}

# ── ECR repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "model" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# ── SageMaker execution role ──────────────────────────────────────────────────
resource "aws_iam_role" "sagemaker" {
  name = "${var.project_name}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# TODO: VPC, subnets, security groups if needed
