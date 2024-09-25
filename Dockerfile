
FROM python:3.7-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y

EXPOSE 8080

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]