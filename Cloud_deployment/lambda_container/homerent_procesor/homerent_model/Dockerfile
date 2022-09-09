ARG AWS_PYTHON_VERSION_TAG="3.9"
FROM public.ecr.aws/lambda/python:${AWS_PYTHON_VERSION_TAG}
# ARG TASK_DIR="/var/task"
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt -t ${LAMBDA_TASK_ROOT}
ADD *.py ./
RUN mkdir ./b9cf70dad0af4a9fb8db7875b8431947
RUN mkdir ./b9cf70dad0af4a9fb8db7875b8431947/artifacts
RUN mkdir ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/
RUN mkdir ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking
RUN mkdir ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting
RUN ls ./b9cf70dad0af4a9fb8db7875b8431947/*
# COPY b9cf70dad0af4a9fb8db7875b8431947/ ./b9cf70dad0af4a9fb8db7875b8431947
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/MLmodel ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/MLmodel
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/python_env.yaml ./b9cf70dad0af4a9fb8db7875b8431947/stacking/python_env.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/conda.yaml ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/MLmodel ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/MLmodel
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/model.pkl ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/python_env.yaml ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/python_env.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/requirements.txt ./b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt

RUN mkdir ./var
RUN mkdir ./var/task
RUN mkdir ./var/task//b9cf70dad0af4a9fb8db7875b8431947
RUN mkdir ./var/task//b9cf70dad0af4a9fb8db7875b8431947/artifacts
RUN mkdir ./var/task//b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/
RUN mkdir ./var/task//b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking
RUN mkdir ./var/task//b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting
RUN ls ./b9cf70dad0af4a9fb8db7875b8431947/*
# COPY b9cf70dad0af4a9fb8db7875b8431947/ ./b9cf70dad0af4a9fb8db7875b8431947
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/MLmodel ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/MLmodel
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/python_env.yaml ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/python_env.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/conda.yaml ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/conda.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/MLmodel ./var/task/b9cf70dad0af4a9fb8db7875b8431947/stacking/artifacts/models/MLmodel
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/model.pkl ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/model.pkl
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/python_env.yaml ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/python_env.yaml
COPY /b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/voting/requirements.txt ./var/task/b9cf70dad0af4a9fb8db7875b8431947/artifacts/models/stacking/requirements.txt
# COPY ./b9cf70dad0af4a9fb8db7875b8431947/ ./b9cf70dad0af4a9fb8db7875b8431947
RUN ls ./b9cf70dad0af4a9fb8db7875b8431947/*
# COPY b9cf70dad0af4a9fb8db7875b8431947 ./b9cf70dad0af4a9fb8db7875b8431947/
COPY result_scaler.sav ./result_scaler.sav
COPY scaler.sav ./scaler.sav
CMD ["index.lambda_handler"]
