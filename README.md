# AI in Healthcare: Diabetes Prediction

This project aims to predict diabetes in patients using machine learning techniques. The dataset used is the Pima Indians Diabetes Dataset, which contains medical data for female patients of Pima Indian heritage.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction

The goal of this project is to develop a machine learning model to predict whether a patient has diabetes based on various medical features. The project involves data preprocessing, handling missing values, model building with hyperparameter tuning, and model evaluation.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Dataset, which can be found [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

The dataset contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 or 1)

## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

 ```bash
git clone https://github.com/GokulR2003/AI-in-Healthcare.git
cd diabetes-prediction
```

3. Run the script:
```bash
 python diabetes_prediction.py
```

## Model Details
The project uses a RandomForestClassifier with hyperparameter tuning performed using GridSearchCV. The hyperparameters tuned include the number of estimators, maximum depth, minimum samples split, and minimum samples leaf.

## Evaluation
The model's performance is evaluated using:Confusion MatrixAccuracy ScoreClassification ReportROC AUC ScoreCross-validation is also performed to get a more robust estimate of the model's performance.

## Results
The results are displayed in terms of accuracy, precision, recall, F1-score, and ROC AUC score. Visualizations include a heatmap of the confusion matrix and an ROC curve.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
