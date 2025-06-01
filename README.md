# Survival Mortality Prediction System

## Overview

This repository provides a machine learning framework for predicting breast cancer mortality status using clinical data, based on the SEER dataset. Multiple classification models—including Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes—are developed, compared, evaluated, and optimized for predictive accuracy.

## Features

- **Data Preparation**: Loads and explores preprocessed mortality status data.
- **Model Development**: Implements and compares Logistic Regression, KNN, and Naive Bayes classifiers.
- **Model Evaluation**: Utilizes accuracy, confusion matrix, classification report (precision, recall, F1-score), and ROC curve for comprehensive model assessment.
- **Hyperparameter Tuning**: Optimizes KNN classifier using GridSearchCV.
- **Reproducibility**: Employs `train_test_split` with stratification and fixed random state for experimental consistency.

## Notebooks

- `Notebook_2.ipynb`: Main workflow for data loading, model training, evaluation, and optimization.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries (install via pip):

  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

### Data

- The system expects a preprocessed mortality status dataset in CSV format (e.g., `mortality_status.csv`).  
- Update the path as needed in the notebook.

### Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/WDT1203/Survival_MortalityPredictionSystem.git
   cd Survival_MortalityPredictionSystem
   ```

2. Launch `Notebook_2.ipynb` in Jupyter or Colab.

3. Follow the sections for:
   - Data loading and exploration
   - Model development and evaluation
   - Hyperparameter tuning (for KNN)
   - Comparison of model performance

## Model Details

- **Logistic Regression**: Baseline classifier for binary outcome prediction.
- **KNN Classifier**: Both standard and optimized (GridSearchCV) versions included.
- **Naive Bayes**: Gaussian Naive Bayes for probabilistic classification.

## Evaluation Metrics

- **Accuracy**: Percentage of correct predictions on test set.
- **Confusion Matrix**: Visualizes true/false positives and negatives.
- **Classification Report**: Displays precision, recall, and F1-score.
- **ROC Curve**: Plots model discrimination ability.

## Authors

- **Author**: W.Dumindu Tharushika
- **Peer Reviewer**: Lakindu Karunanayake

## License

This project is for academic/research purposes. Please cite appropriately.

---

*For any questions or suggestions, please open an issue or contact the authors.*
```
