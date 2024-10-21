# Credit Card Fraud Detection

## Tyren Leong and Adrian Damian 

## Description

This project uses a Naive Bayes Classifier to classify whether a credit card data instance is fraud or not.

Dataset source: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

>Features of each data instance include:

>>distance_from_home - the distance from home where the transaction happened

>>distance_from_last_transaction - the distance from last transaction happened

>>ratio_to_median_purchase_price - ratio of purchased price transaction to median purchase price

>>repeat_retailer - is the transaction happened from same retailer

>>used_chip - is the transaction through chip (credit card)

>>used_pin_number - is the transaction happened by using PIN number

>>online_order - is the transaction an online order

>>**fraud - is the transaction fraudulent**


## Steps
The data set is processed by checking for missing values then split into 80% for training and 20% for testing.
The Naive Bayes Classifier (NBC) model is trained on the training data.
The model is used to predict both training and testing data.
The evaluation metrics for both training and testing data are computed.

## Results

|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.950635   | 0.593926 | 0.984738  | 0.677395 | 0.303958   |
| Test       |   0.95057    | 0.984842 | 0.984842  | 0.679338 | 0.303189   |


## Installation

Clone the Repository


```bash
git clone https://github.com/tyrenleong/Stock-Portfolio-Optimization-and-Risk-Analysis.git
cd 
```
