# Credit Card Fraud Detection

## Tyren Leong

## Description

This project uses machine learning algorithms to detect fraudulent credit card transactions using Apache Spark MLlib through PySpark and SciPy Scikit-learn. Some data analysis is performed through Spark SQL.

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


## Approach
### Three machine learning approaches were used:
1. Multi-layer Perceptron (MLP) Classifier 
2. Naive Bayes Classifier 
3. Logistic Regression Model 

### PySpark MLlib Steps
- The data set is processed by checking for missing values then split into 80% for training and 20% for testing.
- Features are vectorized using PySpark's VectorAssembler.
- The machine learning algorithms from pyspark.ml.classification are trained on the training data.
- The model is used to predict both training and testing data. (MLP model only predicted testing data)
- The evaluation metrics for both training and testing data are computed.

### SciPy Scikit-learn Steps
- The data set is processed by checking for missing values then split into 80% for training and 20% for testing.
- A Standard Scaler is used to Standardize the features of the training and test data.
- The machine learning models are trained on the training data.
- The model is used to predict both training and testing data.
- The evaluation metrics for both training and testing data are computed.

# Brief Explanatory Data Analysis
Number of rows with fraud 87403
Fraud percentage: 8.7403%
Number of rows without fraud: 912597
Number of rows with fraud and is an online order: 82711
Number of rows with fraud and chip is used: 22410
Number of rows with fraud and is a repeated retailer: 76925

Queried through SQL statements, around 8.7403% of the dataset are fraudulent credit card transactions. Most of the fraudulent cases are online orders and are at repeated retailer.

# Results
## PySpark MLlib

### Multi-layer Perceptron (MLP) Classifier
#### Model 1
>This model uses 2 hidden layers with 32 and 16 neurons respectively. It uses the Sigmoid function as its activation function and Gradient Descent as the solver.

|   Metrics  | Accuracy |
| :--------- | :------: |
| Test       |   0.912364   |

#### Model 2
>This model uses 1 hidden layer with 32 neurons. It uses the Sigmoid function as its activation function and the identity function as the solver.

|   Metrics  | Accuracy |
| :--------- | :------: |
| Test       |   0.917059   |
### Gaussian Naive Bayes (GNB) Classifier 
|   Metrics  | Accuracy | 
| :--------- | :------: | 
| Training   |   0.920065   | 
| Test       |   0.920601   |
### Logistic Regression Model (LOGREG)
|   Metrics  | Accuracy | 
| :--------- | :------: | 
| Training   |   0.958757   | 
| Test       |   0.958599   |




## SciPy Scikit-learn
### Multi-layer Perceptron (MLP) Classifier
#### Model 1
>This model uses 2 hidden layers with 64 and 32 neurons respectively. It uses the Sigmoid function as its activation function and Stochastic Gradient Descent as the solver.

|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.995315   | 0.968821 | 0.997854  | 0.973092 | 0.015563   |
| Test       |   0.995365    | 0.968483 | 0.997935  | 0.973308 | 0.015362   |

#### Model 2
>This model uses 2 hidden layers with 64 and 32 neurons respectively. It uses the Sigmoid function as its activation function and the identity function as the solver.

|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.959433   | 0.610547 | 0.992862  | 0.724667 | 0.133848   |
| Test       |   0.959065    | 0.607988 | 0.992627  | 0.721597 | 0.135543   |

#### Model 3
> This model uses 1 hidden layer with 64 neurons. It uses the Sigmoid function as its activation function and Stochastic Gradient Descent as the solver.

|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.994216   | 0.956842 | 0.997797  | 0.96659 | 0.019166   |
| Test       |   0.99436    | 0.957882 | 0.997847  | 0.967361 | 0.018971   |

### Gaussian Naive Bayes (GNB) Classifier 
|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.95068   | 0.593299 | 0.9848  | 0.677088 | 0.303561   |
| Test       |   0.949905    | 0.593009 | 0.984516  | 0.676692 | 0.307684   |


### Logistic Regression Model (LOGREG)
|   Metrics  | Accuracy | Sensitivity | Specificity | F1 Score | Log Loss |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Training   |   0.960955   | 0.712381 | 0.984753  | 0.761236 | 0.478306   |
| Test       |   0.961695    | 0.720537 | 0.984827  | 0.767051 | 0.463332   |

# Summary
- Generally, Scikit-learn models **'trained much quicker and are more accurate'** than MLlib models. Although PySpark utilizes parallel processing, training time was still slower as it may be due to the relatively small dataset. This means we did not take full advantage of the benefits of PySpark.

- In Scikit-learn, MLP is the most accurate when detecting credit card fraud, flagging less false positives and negatives while also having more confidence when classifying.

- In MLlib, logistic regression had the highest accuracy, beating out the neural network model


## Installation

Clone the Repository


```bash
git clone https://github.com/tyrenleong/Credit-Card-Fraud-Detection.git
cd 
```
