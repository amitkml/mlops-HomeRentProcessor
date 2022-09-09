
# declares a permissions (send logs to CloudWatch, CodeCommit access) for the Lambda function
resource "aws_iam_role" "houserent_predictor_lambda_role" {
  name = "text-summarization-lambda-role"

  tags = {
    group      = var.default_resource_group
    created_by = var.default_created_by
  }

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "houserent_predictor_lambda" {
  role       = aws_iam_role.houserent_predictor_lambda_role.name
  policy_arn = aws_iam_policy.houserent_predictor_lambda_policy.arn
}

resource "aws_iam_policy" "houserent_predictor_lambda_policy" {
  name        = "text-summarization-lambda-policy"
  description = "Policy for the text summarization function to put records into the Amazon Kinesis Data Firehose"
  path        = "/"

  policy = <<-EOF
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "xray:GetSamplingRules",
          "xray:GetSamplingStatisticSummaries",
          "xray:GetSamplingTargets",
          "xray:PutTelemetryRecords",
          "xray:PutTraceSegments",
          "ecr:*"
        ],
        "Resource": [
          "*"
        ]
      },
       {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "es:DeleteOutboundConnection",
                "es:DeletePackage",
                "es:ListElasticsearchInstanceTypeDetails",
                "es:ListDomainsForPackage",
                "ec2:CreateNetworkInterface",
                "ec2:DescribeInstances",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DeleteNetworkInterface",
                "ec2:AttachNetworkInterface",
                "s3:*",
                "sqs:*",
                "es:ListInstanceTypeDetails",
                "es:AcceptInboundConnection",
                "es:DeleteElasticsearchServiceRole",
                "es:DescribeInboundConnections",
                "es:DescribeOutboundConnections",
                "es:DescribeReservedInstances",
                "es:AcceptInboundCrossClusterSearchConnection",
                "es:DescribeReservedInstanceOfferings",
                "es:DescribeInstanceTypeLimits",
                "es:DeleteInboundCrossClusterSearchConnection",
                "es:DescribeOutboundCrossClusterSearchConnections",
                "es:DeleteOutboundCrossClusterSearchConnection",
                "es:DescribeReservedElasticsearchInstanceOfferings",
                "es:CreateServiceRole",
                "es:CreateElasticsearchServiceRole",
                "es:UpdatePackage",
                "es:RejectInboundCrossClusterSearchConnection",
                "es:DeleteInboundConnection",
                "es:GetPackageVersionHistory",
                "es:RejectInboundConnection",
                "es:PurchaseReservedElasticsearchInstanceOffering",
                "es:DescribeInboundCrossClusterSearchConnections",
                "es:ListVersions",
                "es:DescribeReservedElasticsearchInstances",
                "es:ListDomainNames",
                "es:PurchaseReservedInstanceOffering",
                "es:CreatePackage",
                "es:DescribePackages",
                "es:ListElasticsearchInstanceTypes",
                "es:ListElasticsearchVersions",
                "es:DescribeElasticsearchInstanceTypeLimits"
            ],
            "Resource": "*"
        }
    ]
  }
  EOF
}
