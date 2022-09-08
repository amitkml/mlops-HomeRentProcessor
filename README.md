# Project Problem statement

For the project, we will ask you to build an end-to-end ML project. For that, you will need:

- Select a dataset that you're interested in (see datasets.md)
- Train a model on that dataset tracking your experiments
- Create a model training pipeline
- Deploy the model in batch, web service or streaming
- Monitor the performance of your model
- Follow the best practices

# Technologies

- Cloud: AWS
- Experiment tracking tools: Weights & Biases
- Workflow orchestration: Prefect
- Monitoring: Evidently
- CI/CD: Github actions
- Infrastructure as code (IaC): Terraform

# Project Quality Gate
- Problem description
  - Problem is well described and it's clear what the problem the project solves
- Cloud
  - points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure
- Experiment tracking and model registry
  - Both experiment tracking and model registry are used
- Workflow orchestration
  - Fully deployed workflow
 - Model deployment
  - The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
- Model monitoring
  - Comprehensive model monitoring that send alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
- Reproducibility
  - Instructions are clear, it's easy to run the code, and the code works. The version for all the dependencies are specified.
- Best practices
  - There are unit tests
   - There is an integration test
   - Linter and/or code formatter are used
   - There's a Makefile
   - There are pre-commit hooks
   - There's a CI/CI pipeline






