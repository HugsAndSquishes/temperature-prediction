# 🌡️ Surface Temperature Prediction Project

## 📊 Dataset

The data comes from Kaggle and includes monthly average surface temperatures from 1940-2024. You can download it here:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/samithsachidanandan/average-monthly-surface-temperature-1940-2024/data)

After downloading, place the CSV file here in the root directory:

```
/average-monthly-surface-temperature.csv
```

## 🛠️ Setup

1. Clone this repo
2. Create a virtual environment (highly recommended!)
   ```
   python -m venv temp_env
   source temp_env/bin/activate  # On Windows: temp_env\Scripts\activate
   ```
3. Install the dependencies
   ```
   pip install -r requirements.txt
   ```

## 🔄 Project Workflow

The project has a specific order to run things:

### 1. Data Preprocessing

First, run the preprocessing notebook to clean, validate, and prepare the data for modeling:

```
jupyter notebook notebooks/preprocessing.ipynb
```

This will generate processed datasets in the `/data/processed` directory.

### 2. Model Training & Evaluation

We have several modeling approaches:

#### Linear Regression & LSTM (Notebooks)

Run these notebooks in this order:

```
jupyter notebook notebooks/modeling.ipynb  # Linear regression models
jupyter notebook notebooks/LSTM_comparison.ipynb  # Neural network approach
```

#### Other Models (Python Scripts)

These can be run after the preprocessing: These are that dont require preprocessing files.

```
python scripts/RandomeForest.py  # Random Forest regression
python scripts/GradBoost.py      # Gradient Boosting regression
python scripts/TimeSeries.py     # Time series models (ARIMA, SARIMA)
```

## 📈 Evaluation Metrics

All models' performance metrics are stored in the `/metrics` directory for comparison. We track:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 📁 Project Structure

```
/
├── data/
│   ├── raw/                # Raw dataset from Kaggle
│   └── processed/          # Cleaned & feature-engineered data
├── notebooks/              # Jupyter notebooks for analysis
│   ├── preprocessing.ipynb # Data cleaning and feature engineering
│   ├── modeling.ipynb      # Linear regression models
│   ├── LSTM_comparison.ipynb # Deep learning models
│   └── LSTM_pipeline.py    # LSTM helper functions
├── scripts/                # Standalone model implementations
│   ├── RandomeForest.py    # Random Forest implementation
│   ├── GradBoost.py        # Gradient Boosting implementation
│   └── TimeSeries.py       # ARIMA and SARIMA models
├── metrics/                # Performance metrics for all models
├── requirements.txt        # Project dependencies
└── README.md               # This file!
```

## 🤔 Troubleshooting

- Make sure your Python version is 3.8+ for compatibility with all packages
