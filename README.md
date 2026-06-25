# adult-income-prediction
Machine learning project predicting whether annual income exceeds $50K using classification models, feature engineering, and hyperparameter tuning.
# Adult Income Prediction

## Overview

This project predicts whether an individual's annual income exceeds $50,000 using machine learning classification models.

The objective is to identify the demographic, educational, and employment-related factors most strongly associated with higher income levels and evaluate the performance of different predictive models.

## Business Problem

Income prediction can support a variety of business applications, including:

* Credit risk assessment
* Customer segmentation
* Financial planning
* Lending decisions
* Marketing strategy development

This project frames income prediction as a binary classification problem:

* <=50K
* > 50K

## Dataset

**Dataset:** UCI Adult Income Dataset

The dataset contains demographic, educational, employment, and financial information for working adults.

### Selected Features

#### Numerical Features

* Age
* Education Number
* Hours Worked Per Week
* Capital Gain
* Capital Loss

#### Categorical Features

* Sex
* Workclass
* Education
* Marital Status
* Occupation

## Data Preparation

### Cleaning

* Removed missing values
* Removed duplicate records

### Feature Engineering

* One-hot encoding for categorical variables
* Standardisation for numerical features where appropriate
* Train-test split for model evaluation

## Exploratory Data Analysis

Key observations included:

* Higher-income individuals were predominantly male
* Income generally increased with age
* Capital gain was strongly associated with higher income levels
* Individuals working more hours per week were more likely to earn above $50K

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model.

### Decision Tree

Used to capture non-linear relationships between variables.

### Random Forest

Selected as the primary model due to its strong predictive performance and ability to handle complex feature interactions.

### Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to optimise the Random Forest model.

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 85%      |
| Random Forest       | 84%      |
| Decision Tree       | 82%      |

### Tuned Random Forest

* Cross-validation Accuracy: 85.95%
* Test Accuracy: 86%

The tuned Random Forest demonstrated strong predictive performance and good generalisation to unseen data.

## Feature Importance

The most influential features included:

* Marital Status
* Capital Gain
* Age
* Education Level
* Hours Worked Per Week

These variables showed the strongest relationship with income prediction.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Random Forest
* Logistic Regression
* Decision Tree
* GridSearchCV
* Matplotlib
* Seaborn

## Skills Demonstrated

* Machine Learning
* Classification Modelling
* Feature Engineering
* Hyperparameter Tuning
* Model Evaluation
* Feature Importance Analysis
* Data Visualisation

## Author

Arissa Andrade Araujo Santos

Higher Diploma in Data Analytics – First Class Honours
