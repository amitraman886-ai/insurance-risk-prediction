
#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load dataset
df = pd.read_csv("insurance.csv")

## Create target column
df['risk'] = df['charges'].apply(lambda x: 1 if x > 15000 else 0)

#Convert text data to numeric
df['sex'] = df['sex'].map({'male':0, 'female':1})
df['smoker'] = df['smoker'].map({'no':0, 'yes':1})

# Select features and target
X = df[['age','sex','bmi','children','smoker']]
y = df['risk']

## Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train model
model = LogisticRegression()
model.fit(X_train, y_train)

#Make prediction
pred = model.predict(X_test)

#Check accuracy
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

#Compare predicted vs actual
print("First 5 predictions:", pred[:5])
print("Actual values:", y_test[:5].values)

