# Default Tags
variable "default_resource_group" {
  description = "Default value to be used in resources' Group tag."
  default     = "mlops-zoomcamp"
}

variable "lambda_cloudwatch_logs_retention_days" {
  type        = number
  description = "The number of days which Lambda CloudWatch logs should be retained"
  default     = 60
}


variable "default_created_by" {
  description = "Default value to be used in resources' CreatedBy tag."
  default     = "akayal"
}

# AWS Settings
# variable "aws_region" {
#   default = "us-east-1"
# }

variable "aws_profile" {
  default = "amit_vs_user"
}

# variable "function_version" {
#   description = "Version of the text_summarization Lambda function to use."
# }

# variable "s3_source_code_bucket" {
#   description = "Name of the S3 bucket hosting the code for text_summarization python notebook."
# }

variable "s3_input_data_bucket" {
  description = "New bucket for storing the input file"
  default     = "mlops-zoomcamp-home-rent-input"
}

variable "s3_output_data_bucket" {
  description = "New bucket for storing output file by lambda based model."
  default     = "mlops-zoomcamp-home-rent-output"
}
