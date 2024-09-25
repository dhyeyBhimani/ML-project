FROM python:3.7-slim-buster
WORKDIR /ML-project
COPY . /ML-project

RUN apt update -y

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]
