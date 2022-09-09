# Best Practices
Following best practices been deployed into the code.

- Testing Python code with pytest
- Integration tests with docker-compose
- Testing cloud services with LocalStack
- Code quality: linting and formatting
- Git pre-commit hooks
- Makefiles and make

## Code Quality
- The git root directory has https://github.com/amitkml/mlops-HomeRentProcessor/blob/main/.pre-commit-config.yaml file which has default_stages: [commit, push] to ensure the validation always triggered.
- The https://github.com/amitkml/mlops-HomeRentProcessor/blob/main/pyproject.toml enforces other rules related to quality
- make file has all associated commands to ensure build and test

## Installation and Reproducability

- First you need to download the release https://github.com/amitkml/mlops-HomeRentProcessor/releases/tag/Final_Model which has the trained model and weights. Please ensure that this is kept in Cloud_deployment\lambda_container\homerent_procesor\homerent_model as the docker command will expect this.
- The make integration_test will trigger all associated steps to build the docker container and trigger integration test.
