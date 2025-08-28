## Implied-volatility-predictor
This project focuses on predicting the implied volatility (IV) of Ethereum (ETH) options. By leveraging high-frequency order book data and cross-asset information from Bitcoin (BTC), this model aims to provide accurate short-term IV forecasts. This README provides a comprehensive overview of the project, from feature engineering to model training and evaluation.

## Features
This model's predictive power is built upon a robust set of engineered features designed to capture the complex dynamics of the cryptocurrency market. These features provide a multi-faceted view of market activity, enabling more accurate IV predictions.

#Core Features
1)Realized Volatility: To capture recent price fluctuations, we calculate the rolling realized volatility over 10-second and 60-second windows. This is derived from log returns of the mid-price, providing a dynamic measure of market risk. 📈

2)Ask-Bid Spread: The spread between the best ask and bid prices is a key indicator of market liquidity and transaction costs. We compute the spread at multiple levels of the order book (levels 1 and 3) to gain a deeper understanding of market depth.

3)Weighted Average Price (WAP): The WAP is calculated for levels 1 and 5 of the order book. This metric provides a more comprehensive view of the asset's price by factoring in both price and volume, offering a more stable price indicator than the mid-price alone.

4)Order-Book Imbalance (OBI): OBI measures the directional pressure in the order book by comparing the volume of buy and sell orders. This is calculated for levels 1 and 5 and serves as a powerful predictor of short-term price movements.

#Advanced Features
1)Lagged Features: To incorporate the influence of past market states on future volatility, we introduce lagged versions of key features. Specifically, we use 1-second, 5-second, and 10-second lags for both OBI and WAP, allowing the model to learn from historical patterns.

2)ross-Asset Volatility: Given the high correlation between ETH and BTC, we incorporate the realized volatility of BTC as a cross-asset feature. This is calculated over a 60-second window and provides valuable context about the broader market sentiment. Lagged versions of this feature at 5-second and 10-second intervals are also included to capture delayed market reactions.

#Machine Learning Model
At the heart of this project is the LightGBM (Light Gradient Boosting Machine), a high-performance gradient boosting framework. LightGBM is well-suited for this task due to its ability to handle large datasets with high efficiency and accuracy.

Model Training and Validation
Cross-Validation: To ensure the model's robustness and prevent overfitting, we employ TimeSeriesSplit for cross-validation. This technique is specifically designed for time-series data, as it preserves the temporal order of observations. The data is split into 7 consecutive folds, with the model being trained on past data and validated on future data in each fold.

Hyperparameters: The LightGBM model is configured with the following key hyperparameters:
n_estimators: 105
learning_rate: 0.023
objective: 'regression'
metric: 'rmse'

Feature Selection: After an initial round of training, a feature importance analysis is conducted to identify the most influential features. The model is then retrained using only the top 20 features, which helps to reduce noise and improve generalization.

Project Structure
The project is organized into the following key components:

goquantfinalsubmission.ipynb: The main Jupyter notebook containing the complete workflow, including data loading, feature engineering, model training, and submission generation.

ETH_train.csv / ETH_test.csv: Training and testing datasets for Ethereum.

BTC_train.csv / BTC_test.csv: Training and testing datasets for Bitcoin, used for cross-asset feature engineering.

submission_final.csv: The final output file containing the predicted implied volatility values.

Setup and Usage
To get started with this project, follow these steps:

Prerequisites
Ensure you have the following libraries installed:

pandas

numpy

scikit-learn

lightgbm

matplotlib

Running the Code
Clone the repository to your local machine.

Place the dataset files (ETH_train.csv, ETH_test.csv, BTC_train.csv, BTC_test.csv) in the specified path within the notebook.

Open and run the goquantfinalsubmission.ipynb notebook. The notebook is structured to execute the entire pipeline, from data preprocessing to generating the final submission file.

Results
The model's performance is evaluated using several key metrics, which are calculated for each fold of the cross-validation process. The final results demonstrate the model's effectiveness in forecasting implied volatility.

Performance Metrics
Root Mean Squared Error (RMSE): The average RMSE across all folds is approximately 0.000047.

Mean Absolute Error (MAE): The average MAE is approximately 0.000033.

R-squared (R²): The R-squared score for the final fold is 0.466, indicating that the model explains a significant portion of the variance in the data.

Pearson Correlation: The average Pearson correlation coefficient is 0.612, signifying a strong positive relationship between the predicted and actual IV values.

Visualizations
Actual vs. Predicted IV: A line graph comparing the actual and predicted IV for the last 5,000 data points of the final validation fold provides a visual confirmation of the model's tracking ability.

Feature Importance: A bar chart of feature importances highlights the most influential predictors, offering insights into the key drivers of implied volatility. 📊

This comprehensive approach, combining advanced feature engineering with a powerful machine learning model, results in a robust and accurate solution for implied volatility forecasting.
