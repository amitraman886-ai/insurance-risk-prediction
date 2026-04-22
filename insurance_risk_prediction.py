# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("insurance.csv")

# Create target column
df['risk'] = df['charges'].apply(lambda x: 1 if x > 15000 else 0)

# Convert categorical data to numeric
df['sex'] = df['sex'].map({'male':0, 'female':1})
df['smoker'] = df['smoker'].map({'no':0, 'yes':1})

# One-hot encode 'region' (important missing step)
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Select features and target
X = df.drop(['charges', 'risk'], axis=1)
y = df['risk']

# Train-test split (added random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (increase max_iter to avoid convergence issues)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, prob))

# Compare predicted vs actual
print("\nFirst 5 predictions:", pred[:5])
print("Actual values:", y_test[:5].values)

# Coefficient interpretation
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nModel Coefficients:\n", coefficients)

# Insight note
print("\nNote: Higher positive coefficients increase the probability of being high-risk (1).")
