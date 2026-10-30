# ==========================================
# Diabetes Prediction Using Neural Networks
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load dataset directly from diabetes.csv
df = pd.read_csv("diabetes.csv")
print("Original dataset shape:", df.shape)

# 2. Data Cleaning & Class Balancing
zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df["zero_count"] = (df[zero_features] == 0).sum(axis=1)

df_0 = df[df["Outcome"] == 0]  # Majority class
df_1 = df[df["Outcome"] == 1]  # Minority class

# Sort by zero count and drop rows with the most missing/zero values
df_0 = df_0.sort_values(by="zero_count", ascending=False).iloc[232:]

df_0 = df_0.drop(columns=["zero_count"])
df_1 = df_1.drop(columns=["zero_count"])

# Combine and shuffle to create a balanced dataset in-memory
df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset balanced! New class distribution:")
print(df["Outcome"].value_counts())

# 3. Impute remaining zeros with column medians
for col in zero_features:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)

# 4. Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# 5. Train/Test Split with Stratification
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# 6. Build Neural Network Architecture
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 8. Model Evaluation & Performance Metrics
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
