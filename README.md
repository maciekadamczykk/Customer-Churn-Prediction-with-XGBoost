# Customer Churn Prediction with XGBoost

## Overview
This project aims to predict customer churn using machine learning techniques, specifically the XGBoost algorithm. The workflow includes data cleaning, exploratory data analysis (EDA), model training, evaluation, and visualization of results.
## Modeling Approach & Dataset

The model was intentionally designed to be more conservative, prioritizing higher recall. This means it aims to identify as many potential churners as possible, even if it results in more false positives. In this business context, it is preferable to proactively flag customers at risk of churning rather than miss individuals who may actually leave. This approach enables targeted retention strategies and minimizes the risk of losing valuable customers.

The dataset used for training and evaluation is the Telco Customer Churn dataset from Kaggle, which provides a comprehensive set of customer attributes and churn labels suitable for predictive modeling.

## Project Structure
```
├── data_cleaning.py           # Script for cleaning and preprocessing raw data
├── EDA.ipynb                  # Jupyter notebook for exploratory data analysis
├── evaluate_model.py          # Script for evaluating the trained model
├── main.py                    # Main script for running the pipeline
├── data/
│   ├── processed_data.csv     # Cleaned and processed data
│   └── rawdata.csv            # Original raw data
├── figures/
│   ├── confusion_matrix_custom_threshold.png
│   └── roc_curve.png          # Visualizations of model performance
├── models/
│   ├── final_xgboost_model.joblib  # Saved XGBoost model
│   └── ordinal_encoder.pkl         # Saved encoder for categorical features
```

## Getting Started
### Prerequisites
- Python 3.7+
- Required packages (install via pip):
  - xgboost
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - joblib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/maciekadamczykk/Customer-Churn-Prediction-with-XGBoost.git
   cd Customer-Churn-Prediction-with-XGBoost
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install packages listed above manually)*

## Usage

1. **Exploratory Data Analysis**
    - Open and run `EDA.ipynb` in Jupyter Notebook to explore the raw data and visualize insights.

2. **Data Cleaning**
    - Run `data_cleaning.py` to preprocess raw data:
       ```bash
       python data_cleaning.py
       ```
    - Output: `data/processed_data.csv`

3. **Model Training & Evaluation**
    - Run `main.py` to train the XGBoost model and save artifacts:
       ```bash
       python main.py
       ```
    - Run `evaluate_model.py` to evaluate the model and generate performance metrics/plots:
       ```bash
       python evaluate_model.py
       ```
    - Visualizations are saved in the `figures/` directory.

## Outputs
- **Processed Data:** `data/processed_data.csv`
- **Model Artifacts:** `models/final_xgboost_model.joblib`, `models/ordinal_encoder.pkl`
- **Figures:** ROC curve, confusion matrix, etc. in `figures/`

## Customization
- Modify scripts to adjust preprocessing, model parameters, or evaluation metrics as needed.
- Update `main.py` and `evaluate_model.py` for custom thresholds or additional metrics.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
maciekadamczykk
