resource "aws_s3_bucket" "s3_home_loan_data_input" {
  bucket = "${var.s3_input_data_bucket}-${var.aws_region}"
  acl    = "private"

  # server_side_encryption_configuration {
  #   rule {
  #     apply_server_side_encryption_by_default {
  #       sse_algorithm = "AES256"
  #     }
  #   }
  # }

  tags = {
    description = "Bucket keeping the code for houserent_predictor input dataset"
    group       = var.default_resource_group
    created_by  = var.default_created_by
  }
}


resource "aws_s3_bucket" "s3_home_loan_data_output" {
  bucket = "${var.s3_output_data_bucket}-${var.aws_region}"
  acl    = "private"

  # server_side_encryption_configuration {
  #   rule {
  #     apply_server_side_encryption_by_default {
  #       sse_algorithm = "AES256"
  #     }
  #   }
  # }

  tags = {
    description = "Bucket keeping the code for houserent_predictor output."
    group       = var.default_resource_group
    created_by  = var.default_created_by
  }
}
