# AI-Powered Network Intrusion Detection and Cyber Attack Classification System

## 1. Project Overview

The AI-Powered Network Intrusion Detection and Cyber Attack Classification System is a machine learning-based cybersecurity application designed to detect and classify network activity into different categories.

The system uses the NIDS-28 dataset and applies data preprocessing, exploratory data analysis, feature engineering, feature selection, machine learning classification, model evaluation, hyperparameter tuning, and deployment through a Flask web application and REST API.

## 2. Problem Statement

Network traffic can contain normal activity as well as different types of cyber attacks. Manually identifying these attacks from network traffic is difficult and time-consuming.

This project develops a machine learning system that automatically analyzes network traffic features and classifies the activity into:

- Normal
- DoS
- Probe
- R2L
- U2R

## 3. Objectives

- Analyze network traffic data.
- Clean and preprocess the dataset.
- Perform exploratory data analysis.
- Create useful engineered features.
- Select important features for classification.
- Train multiple machine learning models.
- Compare model performance.
- Apply cross-validation and hyperparameter tuning.
- Develop a final machine learning pipeline.
- Deploy the model using Flask.
- Provide a REST API for predictions.
- Provide a web interface for network intrusion classification.

## 4. Dataset

**Dataset:** NIDS-28 - Multi-Class Network Intrusion Detection Dataset

- Records: 28,000
- Original columns: 35
- Target column: `attack_type`
- Number of target classes: 5

### Target Classes

| Class | Description |
|---|---|
| Normal | Normal network activity |
| DoS | Denial of Service |
| Probe | Network scanning/probing activity |
| R2L | Remote to Local attack |
| U2R | User to Root attack |

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS
- JavaScript
- Joblib
- Jupyter Notebook

## 6. Machine Learning Workflow

The project follows this workflow:

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Categorical Encoding
↓
Feature Selection
↓
Train-Test Split
↓
Feature Scaling
↓
Model Training
↓
Model Evaluation
↓
Cross-Validation
↓
Hyperparameter Tuning
↓
Final Model Pipeline
↓
Model Deployment
↓
Flask Web Application + REST API

## 7. Data Preprocessing

The dataset was checked for:

- Missing values
- Duplicate records
- Constant features
- Numerical and categorical features
- Skewed numerical features
- Potential outliers

No missing values or duplicate records were found.

Three constant features were excluded:

- `land`
- `urgent`
- `is_host_login`

The derived `binary_label` column was also excluded because the project performs five-class classification using `attack_type`.

## 8. Feature Engineering

Two new features were created:

### Total Bytes

```text
total_bytes = src_bytes + dst_bytes