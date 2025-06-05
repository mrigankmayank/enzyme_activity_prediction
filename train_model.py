# train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 500

data = {
    'temperature': np.random.uniform(20, 70, n_samples),       # in Celsius
    'pH': np.random.uniform(4, 9, n_samples),                  # pH range
    'substrate_concentration': np.random.uniform(0.1, 2.0, n_samples), # mol/L
    'inhibitor_concentration': np.random.uniform(0.0, 0.5, n_samples), # mol/L
}

# Assume enzyme activity depends on all the above
df = pd.DataFrame(data)
df['enzyme_activity'] = (
    50 + 
    2 * df['temperature'] - 
    5 * (df['pH'] - 7)**2 +
    20 * df['substrate_concentration'] -
    30 * df['inhibitor_concentration'] +
    np.random.normal(0, 10, n_samples) # noise
)

# 2. Save synthetic dataset (for reference)
df.to_csv('synthetic_dataset.csv', index=False)

# 3. Prepare data
X = df[['temperature', 'pH', 'substrate_concentration', 'inhibitor_concentration']]
y = df['enzyme_activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save model
with open('enzyme_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'enzyme_model.pkl'. Dataset saved as 'synthetic_dataset.csv'.")
