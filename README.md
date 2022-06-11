# Ponzi Scheme Detector and Predictor on Ethereum

## Motivation
Ponzi Scheme in Ethereum has obvious patterns when created and used. We try to utilize machine learning methods to detect the ponzi contract and predict the price change.


## Goal

1. Detect ponzi scheme on ethereum contracts;

2. Predict the burst of the target ponzi scheme.


## Ponzi definition

1. The contract distributes money among investors
2. The contract receives money only from investors
3. Each investor makes a profit if enough investors invest enough money in the contract afterwards
4. The later an investor joins the contract, the greater the risk of losing his investment.


## Model

- Ponzi
  - stats
  - bytecode
    - settings of train/validate
    - preprocess logic
      - composition features
      - how to handle imbalanced data
    - methods
      - Random Forest
      - xgboost
      - NLP NN
    - Result
      - Importance of features  
      - Validation score


## Acknowledgement

We refer the following Github Repos and the following papers.

- https://github.com/WaterSo0910/ponzi-detector
- https://github.com/rshwndsz/ponzi-detector
- https://github.com/sendrosa/ponzi_ethereum
- https://github.com/jmancini96/Crypto_Price_Predictions

  
