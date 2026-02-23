# Credit Card Fraud Detection using AutoEncoders

## Project Overview
This project implements an anomaly detection system to identify fraudulent credit card transactions. Using the **PyOD** library and **TensorFlow**, we employ a Deep Learning **AutoEncoder** to learn the patterns of normal transactions and flag those with high reconstruction errors as potential fraud.

## Dataset
The model uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized transactions made by European cardholders.

## Installation & Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/bibekitani-git/Hands-On-Assignment-4.git

   Install dependencies:

Bash
pip install -r requirements.txt
Place creditcard.csv in the root directory.

Run the script:

Bash
python3 fraud_detection.py
