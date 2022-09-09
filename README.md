# MLOps Zoomcamp Project - Project Guideline

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

# MLOPS - Project - Homerent Prediction

## About Dataset
### Context
Housing in India varies from palaces of erstwhile maharajas to modern apartment buildings in big cities to tiny huts in far-flung villages. There has been tremendous growth in India's housing sector as incomes have risen. The Human Rights Measurement Initiative finds that India is doing 60.9% of what should be possible at its level of income for the right to housing.
Renting, also known as hiring or letting, is an agreement where a payment is made for the temporary use of a good, service, or property owned by another. A gross lease is when the tenant pays a flat rental amount and the landlord pays for all property charges regularly incurred by the ownership. Renting can be an example of the sharing economy.

### Content
In this Dataset, we have information on almost 4700+ Houses/Apartments/Flats Available for Rent with different parameters like BHK, Rent, Size, No. of Floors, Area Type, Area Locality, City, Furnishing Status, Type of Tenant Preferred, No. of Bathrooms, Point of Contact.

### Dataset Glossary (Column-Wise)
- BHK: Number of Bedrooms, Hall, Kitchen.
- Rent: Rent of the Houses/Apartments/Flats.
- Size: Size of the Houses/Apartments/Flats in Square Feet.
- Floor: Houses/Apartments/Flats situated in which Floor and Total Number of Floors (Example: Ground out of 2, 3 out of 5, etc.)
- Area Type: Size of the Houses/Apartments/Flats calculated on either Super Area or Carpet Area or Build Area.
- Area Locality: Locality of the Houses/Apartments/Flats.
- City: City where the Houses/Apartments/Flats are Located.
- Furnishing Status: Furnishing Status of the Houses/Apartments/Flats, either it is Furnished or Semi-Furnished or Unfurnished.
- Tenant Preferred: Type of Tenant Preferred by the Owner or Agent.
- Bathroom: Number of Bathrooms.
- Point of Contact: Whom should you contact for more information regarding the Houses/Apartments/Flats.
