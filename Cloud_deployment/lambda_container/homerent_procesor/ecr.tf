
## creates an ECR registry where Terraform will save Docker container image, which will be later used by out Lambda function
resource "aws_ecr_repository" "repo" {
  name = local.ecr_repository_name
}

# Output ECR Repository URL
output "repo_url" {
  value = aws_ecr_repository.repo.repository_url
}

# allows us to query information about published Docker image
data "aws_ecr_image" "lambda_image" {
  depends_on = [
    null_resource.ecr_image
  ]
  repository_name = local.ecr_repository_name
  image_tag       = local.ecr_image_tag
  # image_tag_mutability = "IMMUTABLE"
}

## This is being done as per https://stackoverflow.com/questions/68658353/push-docker-image-to-ecr-using-terraform
resource "aws_ecr_repository_policy" "demo-repo-policy" {
  repository = aws_ecr_repository.repo.name
  policy     = <<EOF
  {
    "Version": "2008-10-17",
    "Statement": [
      {
        "Sid": "adds full ecr access to the demo repository",
        "Effect": "Allow",
        "Principal": "*",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetLifecyclePolicy",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart"
        ]
      }
    ]
  }
  EOF
}

#  aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
#  cd ${path.module}/homerent_model
# docker build -f Dockerfile -t ${aws_ecr_repository.repo.repository_url}:${local.ecr_image_tag} .
# docker push ${aws_ecr_repository.repo.repository_url}:${local.ecr_image_tag}
# is used to build Docker container and push it to the ECR registry,
# triggers checks changes in the Lambda function code and Dockerfile and allows Terraform understand when to rebuild the image and update the Lambda function


resource "null_resource" "ecr_image" {
  triggers = {
    python_file = md5(file("${path.module}/homerent_model/index.py"))
    docker_file = md5(file("${path.module}/homerent_model/Dockerfile"))
  }

  provisioner "local-exec" {
    command = <<EOT
    cd homerent_model
    (Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
    docker login --username AWS -p $(aws ecr get-login-password --region ${var.aws_region} ) ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
    docker build -t ${local.prefix}-demo-lambda-container .
    docker tag ${local.prefix}-demo-lambda-container:latest ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${local.prefix}-demo-lambda-container:latest
    docker push ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${local.prefix}-demo-lambda-container:latest
    EOT

    interpreter = ["PowerShell", "-Command"]

  }

  # provisioner "local-exec" {
  #   command = <<EOT
  #   cd homerent_model
  #   (Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
  #   docker build -t homerent-processor-demo-lambda-container .
  #   docker tag homerent-processor-demo-lambda-container:latest 222401151454.dkr.ecr.us-east-1.amazonaws.com/homerent-processor-demo-lambda-container:latest
  #   docker push 222401151454.dkr.ecr.us-east-1.amazonaws.com/homerent-processor-demo-lambda-container:latest
  #   EOT

  #   interpreter = ["PowerShell", "-Command"]

  # }
}
