# Model Deployment - Cloud

This page explains how I have deployed the model into cloud and exposed as batch ML service.There are two ways to deploy a model.

- Offline - Batch Deployment
- Online - Web Service and Streaming

## Offline - Batch Deployment
Batch deployment is used when we need our predictions at regular interval of time for example, daily, hourly, weekly or monthly. We have a database with all our data. We have a scoring job, which has our model. It pulls the data from database and applies the model to return the predictions at regular interval of time.

### Solution Architecture

Have followed the architecture as shown below.

![im](https://miro.medium.com/max/492/1*zKD4_zfSVaQLl_MLK9NJEw.png)

## Installation Process

Lambda based docker image has been followed from https://hands-on.cloud/terraform-deploy-python-lambda-container-image/. main.tf has access key and secrect key details and same needs to be updated with proper values. The user should have permission for ECR publish and lambda creation.

**Tasks:**

- ML Development
- Deploy a Real-time lambda based batch model
- Serverless Development of the Model
- Lambda and IAM role Infrastructure deployment
- Inspect the performance in cloudwatch

**Things covered so far:**

- S3 bucket for training data
- S3 bucket output file place
- Lambda function for using model and make prediction
- Sage maker instance for training
- Required IAM role and policy for interaction
- ECR Repo for lambda docker image storage
- Deployment and testing
- Lambda function for profilling and monitoring

**Steps to run:**

- Go inside terraform directory cd terraform
- Run terraform init
- Run terraform plan
- Verify the resources in plan
- Run terraform apply


## References
- https://onema.io/blog/scikit-learn-layer-for-aws-lambda/
- https://www.oneworldcoders.com/blog/using-terraform-to-provision-amazons-ecr-and-ecs-to-manage-containers-docker
- https://hands-on.cloud/terraform-deploy-python-lambda-container-image/
- https://kevsoft.net/2019/04/26/multi-line-powershell-in-terraform.html
- https://www.devopsschool.com/blog/understanding-local-exec-provisioner-in-terraform/
- https://stackoverflow.com/questions/65576285/docker-login-on-ecr-fails-with-400-bad-request-on-powershell-from-jenkins


