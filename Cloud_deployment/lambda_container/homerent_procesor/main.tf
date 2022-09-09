# provider "aws" {
#   region  = var.aws_region
#   profile = var.aws_profile
#   # version = "~> 2.21"
# }

provider "aws" {
  region = var.aws_region
  #   profile = var.aws_profile
  access_key = "SSS"  ## NEED TO BE PROVIDED WITH ACTUAL VALUE
  secret_key = "5idFsf3W/SS" ## NEED TO BE PROVIDED WITH ACTUAL VALUE
  #   version = "3.46.0"
  version = "~> 3.0"
}

provider "template" {
  # version = "~> 2.1"
}

provider "archive" {
  # version = "~> 1.2"
}

data "aws_caller_identity" "current" {}

# region = data.aws_caller_identity.current.arn.region
# locals {
#   text_summarization_processor_name = "home-rent-predictor-processor"
# }
variable "aws_region" {
  description = "The AWS region to create things in."
  default     = "us-east-1"
}

locals {
  prefix              = "homerent-processor"
  account_id          = data.aws_caller_identity.current.account_id
  ecr_repository_name = "${local.prefix}-demo-lambda-container"
  #  ecr_image_tag       = "latest"
  ecr_image_tag = "latest"
  #  text_summarization_processor_name = "homerent-processor-demo-lambda-container"

}
