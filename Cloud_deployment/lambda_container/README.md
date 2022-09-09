# This Repo contains code and info on how to configure AWS sagemaker and terraform.
Lambda based docker image has been followed from https://hands-on.cloud/terraform-deploy-python-lambda-container-image/

## Tasks
1. ML Development
2. Deploy a Real-time lambda based batch model
3. Serverless Development of the Model
4. Lambda and IAM role Infrastructure deployment
5. Inspect the performance in cloudwatch


## Things covered so far
1. S3 bucket for training data
2. S3 bucket output file place
3. Lambda function for using model and make prediction
4. Sage maker instance for training
5. Required IAM role and policy for interaction
6. ECR Repo for lambda docker image storage
7. Deployment and testing
8. Lambda function for profilling and monitoring

## Steps to run
1. Go inside terraform directory `cd terraform`
2. Run `terraform init`
3. Run `terraform plan`
4. Verify the resources in plan
4. Run `terraform apply`
