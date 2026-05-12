import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

df =pd.read_csv("C:\HR_ANALYTICS\data\HR_Employee_Attrition.csv")

df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


le=LabelEncoder()
cat_cols=df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col]=le.fit_transform(df[col])

# split 
X=df.drop('Attrition',axis=1)
y=df['Attrition']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Train 
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Evaluate
y_pred=model.predict(X_test)
print(f"Accuracy:{accuracy_score(y_test,y_pred):.2f}")
print(classification_report(y_test,y_pred))

# save model 
joblib.dump(model,"C:\HR_ANALYTICS\models\model.pkl")
print("Model saved")
