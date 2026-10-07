import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create a dummy DataFrame
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace"],
    "age": [22, 25, 30, 35, 40, 45, 50],
    "salary": [35000, 42000, 55000, 70000, 85000, 98000, 115000],
}
df = pd.DataFrame(data)

# 2. Features and Target
X = df[["age"]]
y = df["salary"]

# 3. Train Model
model = LinearRegression()
model.fit(X, y)

# 4. Save the trained model
joblib.dump(model, "salary_model.pkl")
print("Model trained and saved as 'salary_model.pkl'")