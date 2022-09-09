# Lambda function declaration itself

resource "aws_lambda_function" "home_rent_event_processor" {
  function_name                  = local.ecr_repository_name
  memory_size                    = 1024
  reserved_concurrent_executions = 3
  timeout                        = 600
  depends_on = [
    null_resource.ecr_image, aws_s3_bucket.s3_home_loan_data_output
  ]
  # role                           = aws_iam_role.text_summarization_lambda_role.arn
  package_type = "Image"
  image_uri    = "${aws_ecr_repository.repo.repository_url}@${data.aws_ecr_image.lambda_image.id}"
  # publish                        = true
  role = aws_iam_role.houserent_predictor_lambda_role.arn
   environment {
    variables = {
    ServiceConfiguration__LOGGING_LEVEL    = "INFO"
    ServiceConfiguration__DEST_BUCKET = aws_s3_bucket.s3_home_loan_data_output.id
    }
  }
  tags = {
    group      = var.default_resource_group
    created_by = var.default_created_by
  }

  # lifecycle {
  #   ignore_changes = [
  #     image_uri,
  #   ]
  # }
}

resource "aws_cloudwatch_log_group" "re-them-prediction-logger-log" {
  name              = local.ecr_repository_name
  retention_in_days = var.lambda_cloudwatch_logs_retention_days
  tags = {
    group      = var.default_resource_group
    created_by = var.default_created_by
  }
}

resource "aws_lambda_alias" "ml-backend-predict-alias" {
  name             = "STABLE"
  description      = "A STABLE version of lambda function"
  function_name    = aws_lambda_function.home_rent_event_processor.function_name
  function_version = aws_lambda_function.home_rent_event_processor.version
}

# # Adding S3 bucket as trigger to my lambda and giving the permissions
resource "aws_s3_bucket_notification" "lambda_s3_aws_cost_rep_pres_url_trigger" {
  depends_on = [
    aws_lambda_permission.lambda-trigger-allow-bucket
  ]
  bucket = "${var.s3_input_data_bucket}-${var.aws_region}"
  lambda_function {
    lambda_function_arn = aws_lambda_function.home_rent_event_processor.arn
    events              = ["s3:ObjectCreated:*"]

  }
}

resource "aws_lambda_permission" "lambda-trigger-allow-bucket" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.home_rent_event_processor.arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.s3_home_loan_data_input.arn
}
