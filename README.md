Water Quality Prediction using Machine Learning Regression Models

This repository contains code for training and spatial prediction of water quality parameters using machine learning regression models. The implemented models leverage spectral data obtained from remote sensors to estimate key water quality indicators such as turbidity, nitrates, and phosphates.

Features

Data Preprocessing: Cleans and prepares spectral and water quality datasets.

Model Training: Implements various regression models, including SVR, RFR, and GBR.

Feature Selection: Identifies the most relevant spectral indices (e.g., NDVI, NDWI) for predicting water quality parameters.

Spatial Prediction: Generates spatial estimates of water quality based on trained models.

Performance Evaluation: Compares model accuracy using standard metrics such as R² and RMSE.

Technologies Used

Python

Scikit-learn

NumPy & Pandas

Geospatial Libraries (e.g., Rasterio, GDAL)

Matplotlib & Seaborn for Visualization

Installation

To run the code, install the required dependencies:

pip install -r requirements.txt

Usage

Prepare the dataset and organize spectral data.

Train regression models using train_model.py.

Evaluate model performance.

Use trained models for spatial prediction in WQ.ipynb.


This project is based on methodologies validated in recent research, utilizing UAV-based remote sensing for water quality assessment.
