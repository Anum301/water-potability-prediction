import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data - NEW WORKING LINK
url = "https://cainvas-static.s3.amazonaws.com/media/user_data/tanmay/water_potability.csv"
df = pd.read_csv(url)
df = df.fillna(df.mean(numeric_only=True))

# 2. Split data
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Save model with joblib
joblib.dump(model, "model.pkl")
print("Model saved successfully!")