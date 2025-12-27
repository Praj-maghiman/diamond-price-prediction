import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# 1. Load and Clean
df = sns.load_dataset('diamonds')
df = df[(df[['x','y','z']] != 0).all(axis=1)]

# 2. Manual Ordinal Mapping
cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# 3. Features Selection (7 features: No depth or table)
features_to_keep = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
X = df[features_to_keep]
y = np.log1p(df['price'])

# 4. Build Compact Model (To keep file size < 25MB)
compact_rf = RandomForestRegressor(
    n_estimators=50, 
    max_depth=12, 
    min_samples_leaf=4, 
    random_state=42, 
    n_jobs=-1
)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', compact_rf)
])

# Train on all data
final_pipeline.fit(X, y)

model_filename = 'diamond_model_v2.pkl'
joblib.dump(final_pipeline, model_filename)

print(f"✅ Model Saved: {model_filename}")