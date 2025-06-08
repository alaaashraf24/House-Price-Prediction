# House Price Prediction

This notebook demonstrates a comprehensive workflow for predicting house prices using machine learning models. The process includes data loading, cleaning, exploratory data analysis, feature engineering, feature selection, model training, evaluation, and analysis of results.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Libraries](#libraries)
3.  [Data Loading and Initial Examination](#data-loading-and-initial-examination)
4.  [Data Cleaning](#data-cleaning)
5.  [Feature Engineering](#feature-engineering)
6.  [Target Variable Analysis and Transformation](#target-variable-analysis-and-transformation)
7.  [Feature Selection](#feature-selection)
8.  [Model Training and Evaluation](#model-training-and-evaluation)
9.  [Model Comparison](#model-comparison)
10. [Feature Importance Analysis](#feature-importance-analysis)
11. [Residual Analysis](#residual-analysis)
12. [Conclusion](#conclusion)

## Introduction

This project aims to build a predictive model for house prices. The process involves standard machine learning steps, focusing on understanding the data, preparing it for modeling, training various regression models, and evaluating their performance.

## Libraries

The following libraries are used in this notebook:

-   pandas: For data manipulation and analysis.
-   numpy: For numerical operations.
-   matplotlib.pyplot: For basic plotting.
-   seaborn: For enhanced data visualization.
-   datetime: For handling date information.
-   warnings: To ignore specific warnings.
-   sklearn: For machine learning tools (model selection, models, preprocessing, metrics, feature selection).
-   scipy: For scientific and technical computing (statistical functions).
-   xgboost: For XGBoost regression (although not explicitly used in the final model comparison loop, the import suggests consideration).

## Data Loading and Initial Examination

The dataset is loaded from a CSV file named data.csv. Basic checks are performed to understand the data's structure, data types, basic statistics, and identify initial data quality issues like duplicates and potentially unrealistic values.

## Data Cleaning

This section focuses on cleaning the raw data. Steps include:

-   Handling invalid (non-positive) house prices.
-   Removing extreme outliers based on a Z-score threshold (e.g., > 3 standard deviations) for key numerical features like price, living area, and lot area.
-   Converting relevant columns to appropriate data types (e.g., 'date' to datetime objects).
-   Intelligently filling missing values (e.g., filling yr_renovated with 0 if missing, assuming no renovation).

## Feature Engineering

New features are created to enhance the model's ability to capture patterns in the data. This includes:

-   *Age-related features*: house_age, age_squared, is_renovated, years_since_renovation.
-   *Size and ratio features*: price_per_sqft, living_to_lot_ratio, total_sqft.
-   *Interaction features*: bed_bath_ratio, total_rooms, sqft_per_bedroom.
-   *Quality score*: Combining view and condition features.
-   *Location-based features*: One-hot encoding for city, calculating city_avg_price, city_price_std, and city_count.
-   *Time-based features*: sale_month, sale_year, sale_quarter, is_summer.
-   *Log transformations*: Applying log1p to skewed features like sqft_living, sqft_lot, and sqft_above to make their distributions more normal.

## Target Variable Analysis and Transformation

The distribution of the target variable (price) is analyzed. Due to its typically right-skewed nature, a log transformation (log_price = log(price)) is applied to make the distribution more symmetrical and closer to normal, which can improve the performance of many regression models. The original and transformed distributions, along with a Q-Q plot, are visualized.

## Feature Selection

Before training models, feature selection is performed to identify the most relevant features and reduce dimensionality.

-   Columns deemed irrelevant or redundant (e.g., original price, date, address details) are excluded.
-   Features with negligible variance are removed.
-   SelectKBest with f_regression is used to select the top k features based on their F-statistic with respect to the target variable. Missing values in features are handled by filling with the median before selection.

## Model Training and Evaluation

The data is split into training and testing sets. Features are scaled using RobustScaler, which is less sensitive to outliers, particularly for linear models.

Several regression models are trained and evaluated:

-   Ridge Regression
-   Lasso Regression
-   Random Forest Regressor
-   Extra Trees Regressor
-   Gradient Boosting Regressor

Each model is trained on the training data (using scaled data for linear models and original data for tree models) and evaluated on the test set. Cross-validation (5-fold) is also performed on the training data to get a more robust estimate of model performance.

Metrics calculated for evaluation include:

-   Mean Squared Error (MSE)
-   Root Mean Squared Error (RMSE)
-   Mean Absolute Error (MAE)
-   R² score (on both the original and log-transformed scales)
-   Cross-validation mean and standard deviation of R² score.

Predictions are transformed back to the original price scale before calculating evaluation metrics that are more interpretable in terms of actual price errors (RMSE, MAE, R²_original).

## Model Comparison

The performance metrics of all trained models are compiled into a DataFrame for easy comparison. The best-performing model is identified based on the R² score on the original price scale.

## Feature Importance Analysis

For tree-based models (Random Forest, Extra Trees, Gradient Boosting), feature importance scores are available. The feature importance is extracted from the best-performing tree model to understand which features contributed most significantly to the price prediction. The top features are printed and visualized.

## Residual Analysis

Residual analysis is performed for the best-performing model. The residuals (difference between actual and predicted prices on the original scale) are plotted against predicted values to check for heteroscedasticity (non-constant variance). A histogram and Q-Q plot of residuals are examined to assess their distribution. An Actual vs Predicted plot is also shown to visually inspect the model's performance across the range of prices.

## Conclusion

The notebook concludes by summarizing the best-performing model and its key evaluation metrics (R², RMSE, and cross-validation results). The residual analysis helps confirm the model's assumptions and identify areas for potential improvement.
