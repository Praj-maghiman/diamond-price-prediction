import seaborn as sns
import pandas as pd
import numpy as np
np.random.seed(42)

df = sns.load_dataset('diamonds')
print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.shape)
print("\n")
#print(df.describe()) #It shows statistical summary of numerical columns

from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
le_cut = LabelEncoder()
df['cut'] = le_cut.fit_transform(df['cut'])
le_color = LabelEncoder()
df['color'] = le_color.fit_transform(df['color'])
le_clarity = LabelEncoder()
df['clarity']  = le_clarity.fit_transform(df['clarity'])
print("After Encoding Process:\n",df.head())

X = df.drop('price', axis=1)
y = df['price']
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size= 0.2, random_state= 42
)

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
model = LinearRegression() # Create
model.fit(X_train, y_train) # Train
y_pred = model.predict(X_test) # Predict
# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_r2 = r2_score(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

print("="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Linear Regression - R²: {r2:.4f}, RMSE: {rmse:.2f}")
print(f"Decision Tree      - R²: {dt_r2:.4f}, RMSE: {dt_rmse:.2f}")
print(f"Random Forest      - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}")
print("="*50)

# Feature Importance for Random Forest
feature_importance = rf_model.feature_importances_ #returns a list
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_values = [r2, dt_r2, rf_r2]
plt.figure(figsize=(10,6))
plt.title('Model Comparison - Diamond Price Prediction')
plt.bar(models, r2_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.ylabel('R^2 Score')
plt.xlabel('Model')
plt.ylim([0.8, 1.0])
plt.savefig('model_comparison.png')

plt.figure(figsize=(10,6))
plt.title('Feature Importance Chart - Diamond Price Prediction')
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#95E1D3')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('feature_importance.png')

rmse_values = [rmse, dt_rmse, rf_rmse]
plt.figure(figsize=(10,6))
plt.bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xlabel('Models')
plt.ylabel('RMSE value')
plt.savefig('rmse_comparison.png')

# Scatter Plot
plt.scatter(y_test, rf_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'r--')
plt.xlabel('ACTUAL')
plt.ylabel('PREDICTED')
plt.title('Prediction Vs Actual Chart')
plt.savefig('prediction_actual.png')

# Residual Plot: Show the errors (difference between actual and predicted).
plt.figure(figsize=(10,6))
residuals = y_test - rf_pred
plt.scatter(rf_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Errors')
plt.savefig('residuals.png')