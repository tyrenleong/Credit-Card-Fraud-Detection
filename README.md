# Credit Card Fraud Detection

## Tyren Leong and Adrian Damian 

## Description

This project uses a Multi-layer Perceptron to classify whether a credit card data instance is fraud or not.

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
- The data set is processed by checking for missing values then split into 80% for training and 20% for testing.
- A Standard Scaler is used to Standardize the features of the training and test data.
- The Multi-layer Perceptron (MLP) model is trained on the training data.
- The model is used to predict both training and testing data.
- The evaluation metrics for both training and testing data are computed.

## Results

### Model 1
This model uses 2 hidden layers with 64 and 32 neurons respectively. It uses the Sigmoid function as its activation function and Stochastic Gradient Descent as the solver.
|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.995315   | 0.968821 | 0.997854  | 0.973092 | 0.015563   |
| Test       |   0.995365    | 0.968483 | 0.997935  | 0.973308 | 0.015362   |

### Model 2
This model uses 2 hidden layers with 64 and 32 neurons respectively. It uses the Sigmoid function as its activation function and the identity function as the solver.
|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.959433   | 0.610547 | 0.992862  | 0.724667 | 0.133848   |
| Test       |   0.959065    | 0.607988 | 0.992627  | 0.721597 | 0.135543   |

### Model 3
This model uses 1 hidden layer with 64 neurons. It uses the Sigmoid function as its activation function and Stochastic Gradient Descent as the solver.
|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.994216   | 0.956842 | 0.997797  | 0.96659 | 0.019166   |
| Test       |   0.99436    | 0.957882 | 0.997847  | 0.967361 | 0.018971   |


## Comparison to NBC
While the results varied greatly depending on the model parameters, all three versions of the model outperformed the Naive Bayes Classifier, with Model 2 outperforming slightly and Model 1 and 3 outperforming greatly. The models had higher Accuracy, Sensitivity, Specificity, and F1 scores, while also having smaller Log Loss values. This shows that the Multi-Layer Perceptron is much more accurate when detecting credit card fraud, flagging less false positives and negatives while also having more confidence when classifying.


## Installation

Clone the Repository


```bash
git clone https://github.com/tyrenleong/Credit-Card-Fraud-Detection.git
cd 
```
