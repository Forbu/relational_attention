# start from simple python image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y curl && apt-get install build-essential -y

RUN pip install pytest torch_scatter
RUN curl -sSL https://install.python-poetry.org | python3 -