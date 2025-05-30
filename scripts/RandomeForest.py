import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
print("Loading dataset...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '../data/raw/average-monthly-surface-temperature.csv')
data = pd.read_csv(csv_path)

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Feature engineering
print("\nPerforming feature engineering...")

# Convert Day to datetime and extract month and year features
data['Day'] = pd.to_datetime(data['Day'])
data['Month'] = data['Day'].dt.month
data['Year_num'] = data['year']  # Numeric year for the model

# Add seasonal features
data['Season'] = data['Month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

# Create country encoding
data = pd.get_dummies(data, columns=['Entity', 'Code'], drop_first=True)

# Prepare features and target
features = ['Year_num', 'Month', 'Season'] + [col for col in data.columns if col.startswith('Entity_') or col.startswith('Code_')]
X = data[features]
y = data['Average surface temperature']

print(f"\nSelected features: {len(features)} features")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, 
                               min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.savefig('actual_vs_predicted.png')
plt.close()

# Visualize feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.savefig('residuals_distribution.png')
plt.close()

# Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Temperature')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.savefig('residuals_vs_predicted.png')
plt.close()



