import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("crop_data.csv")  

soil_types = ["Clayey", "Loamy", "Sandy", "Sandy loam"]  
for soil in soil_types:
    df[f"Soil Type_{soil}"] = (df["Soil Type"] == soil).astype(int)


df = df.drop(columns=["Soil Type"])


X = df.drop(columns=["Crop"])  
y = df["Crop"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

