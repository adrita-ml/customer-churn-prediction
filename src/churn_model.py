import pandas as pd

df = pd.read_csv("data/churn.csv")

print(df.head()) #for displaying first 5 rows
print(df.info()) #for displaying the structure
print(df.describe()) #for displaying mean,min,max of numeric columns
print(df["Churn"].value_counts()) #for displaying how many customers left vs stayed (i.e is dataset balanced or imbalanced)

# to drop unnecessary column
df = df.drop("customerID", axis=1)

#this is a VERY IMPORTANT step, this coverts all text data into numbers as ML needs numbers, this is called Encoding Categorical Data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") #converts TotalCharges to numeric

df = df.dropna() #drop rows with missing values
print(df.isnull().sum()) #this is for checking if there are any missing values present in the dataset

df = pd.get_dummies(df, drop_first=True) #this line is also for Encoding Categorical Data
print(df.head())

#STEP1: Separating features and target (X= input data i.e customer details, y= output i.e will customer leave or not)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

#STEP2: Train-Test Split(model learns with training data, model is evaluated with testing i.e unseen data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

#STEP3: Train Model (we are using Logistic Regression as it is simple and good for binary classification)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=3000, class_weight = 'balanced')
model.fit(X_train, y_train)

#STEP4: Make Predictions
y_pred = model.predict(X_test)

#STEP5: Evaluate the model
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Precision, Recall and F1-score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
