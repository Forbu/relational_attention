# start from simple python image
FROM python:3.9-slim

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install pytest
# insall curl and poetry
RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python3 -