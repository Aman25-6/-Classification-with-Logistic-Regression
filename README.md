 -Classification-with-Logistic-Regression
 Project: Binary Classification with Logistic Regression

This project builds a binary classifier to predict whether a breast cancer diagnosis is benign or malignant using the Breast Cancer Wisconsin dataset. The implementation uses Python with Scikit-learn, Pandas, and Matplotlib.

Objective

The main objective is to apply logistic regression for a binary classification task and evaluate its performance using various metrics. 
Process

1. Dataset: The Breast Cancer Wisconsin dataset from Scikit-learn's built-in datasets was used. [cite: 14]
2. Data Preprocessing: The dataset was split into training and testing sets. [cite_start]The features were then standardized to ensure the model performs accurately.
3.  Model Training: A logistic regression model was trained on the preprocessed data. [cite: 10]
4.  Evaluation: The model's performance was assessed using:
    Confusion Matrix 
    Precision and Recall
    ROC-AUC Score 
5. Threshold Tuning: The impact of adjusting the decision threshold on precision and recall was explored. 

Libraries Used

* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

How to Run**

1.  Clone the repository.
2.  Ensure you have Python and the required libraries installed.
3.  Run the `main.py` script. The script will train the model, print the evaluation metrics, and display the confusion matrix and ROC curve plots.
