output "lambda_name" {
 value = aws_lambda_function.home_rent_event_processor.id
}

output "lambda_name_Arn" {
 value = aws_lambda_function.home_rent_event_processor.arn
}

output "Input_Bucket_Arn" {
 value = aws_s3_bucket.s3_home_loan_data_input.arn
}

output "Output_Bucket_Arn" {
 value = aws_s3_bucket.s3_home_loan_data_output.arn
}
