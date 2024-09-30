# Student Marks Prediction - End-to-End Machine Learning Project


This repository contains an industry-level, end-to-end machine learning project for predicting student marks, developed and deployed on Google Cloud Platform (GCP). The project includes data ingestion, transformation, model training, prediction, and deployment via a Flask application.

# Table of Contents
  
  1.Project Overview
  2.Project Structure
  3.Features
  4.Setup Instructions
  5.How to Run
  6.Deployment on Google Cloud Platform
  7.Technologies Used


# Project Overview

This project predicts student marks based on input data such as gender, race/ethnicity, parental education, lunch type, test preparation course, and scores in reading and writing. The entire machine learning pipeline is built and deployed using Flask and hosted on Google Cloud Platform (GCP) for real-time predictions.

Steps:
  1.Data Ingestion: Reads the dataset and splits it into training and testing datasets.
  2.Data Transformation: Preprocessing of the data, including scaling and encoding.
  3.Model Training: A machine learning model is trained and evaluated on the data.
  4.Prediction: Custom input data is used to predict student marks.
  5.Web Application: A Flask app provides a web interface for users to input data and get predictions.
  6.Deployment: The Flask application is deployed on Google Cloud Platform using Docker and Cloud Run.

# Project Structure
    src/
  ├── components/
  │   ├── data_ingestion.py
  │   ├── data_transformation.py
  │   ├── model_training.py
  ├── exception.py
  ├── logger.py
  ├── pipeline/
  │   ├── predict_pipeline.py
  ├── utils.py
  templates/
  │   ├── index.html
  │   ├── home.html
  app.py
  Dockerfile
  requirements.txt
  README.md

# Features
  1.Data Ingestion: Automatically reads, processes, and splits data into training and testing sets.
  2.Data Transformation: Scales and encodes data for model training.
  3.Model Training: Trains a machine learning model and evaluates it.
  4.Prediction Pipeline: Provides predictions based on user input.
  5.Flask Web App: Real-time predictions via a web interface.
  6.Deployed on GCP: The application is containerized and deployed using Google Cloud Platform for global accessibility.

# Technologies Used
  1.Python
  2.Flask
  3.Pandas
  4.Scikit-learn
  5.Docker
  6.Google Cloud Platform (Cloud Run, Cloud Build)
  7.HTML/CSS for web forms






