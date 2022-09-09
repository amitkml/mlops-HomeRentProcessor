(Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
docker build -t homerent-processor-demo-lambda-container
docker tag homerent-processor-demo-lambda-container:latest 222401151454.dkr.ecr.us-east-1.amazonaws.com/homerent-processor-demo-lambda-container:latest
docker push 222401151454.dkr.ecr.us-east-1.amazonaws.com/homerent-processor-demo-lambda-container:latest
