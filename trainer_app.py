import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Create dummy data for house price prediction
np.random.seed(42)
n_samples = 1000

# Generate features
square_feet = np.random.uniform(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.uniform(0, 50, n_samples)

# Generate target (house prices) with some noise
prices = (
    200 * square_feet +
    50000 * bedrooms +
    35000 * bathrooms -
    1000 * age +
    np.random.normal(0, 50000, n_samples)
)

# Create DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': prices
})

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Print model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R² Score: {train_score:.3f}")
print(f"Testing R² Score: {test_score:.3f}")