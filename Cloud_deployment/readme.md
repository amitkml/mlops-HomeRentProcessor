# Model Deployment - Cloud

This page explains how I have deployed the model into cloud and exposed as batch ML service.There are two ways to deploy a model.

- Offline - Batch Deployment
- Online - Web Service and Streaming

## Offline - Batch Deployment
Batch deployment is used when we need our predictions at regular interval of time for example, daily, hourly, weekly or monthly. We have a database with all our data. We have a scoring job, which has our model. It pulls the data from database and applies the model to return the predictions at regular interval of time.

### Solution Architecture

Have followed the architecture as shown below.

![im](https://miro.medium.com/max/492/1*zKD4_zfSVaQLl_MLK9NJEw.png)

## References
- https://onema.io/blog/scikit-learn-layer-for-aws-lambda/
- https://www.oneworldcoders.com/blog/using-terraform-to-provision-amazons-ecr-and-ecs-to-manage-containers-docker
- https://hands-on.cloud/terraform-deploy-python-lambda-container-image/
- https://kevsoft.net/2019/04/26/multi-line-powershell-in-terraform.html
- https://www.devopsschool.com/blog/understanding-local-exec-provisioner-in-terraform/
- https://stackoverflow.com/questions/65576285/docker-login-on-ecr-fails-with-400-bad-request-on-powershell-from-jenkins
