LOCAL_TAG:=$(shell date +"%Y-%m-%d")
LOCAL_IMAGE_NAME:=batch-model-red-wine-quality:${LOCAL_TAG}
LOCAL_IMAGE_NAME:=batch-homerent-model:${LOCAL_TAG}

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt --no-dependencies
# LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# LOCAL_IMAGE_NAME:=batch-model-red-wine-quality:${LOCAL_TAG}
quality_checks: install

	isort .
	black .
	pylint --recursive=y .


test: quality_checks

	pytest test/ -s

build: test

	docker build -t ${LOCAL_IMAGE_NAME} -f Dockerfile .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME}
	docker run ${LOCAL_IMAGE_NAME} python3 deployment-test-batch-service-mlflow.py
	echo "publishing image ${LOCAL_IMAGE_NAME} to ECR..."
