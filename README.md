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
| Training   |   0.95068   | 0.593299 | 0.9848  | 0.677088 | 0.303561   |
| Test       |   0.949905    | 0.593009 | 0.984516  | 0.676692 | 0.307684   |

![image](https://github.com/user-attachments/assets/879b1acb-49d4-48aa-b12c-4f9ece02fe41)


## Installation

Clone the Repository


```bash
git clone https://github.com/tyrenleong/Credit-Card-Fraud-Detection.git
cd 
```
