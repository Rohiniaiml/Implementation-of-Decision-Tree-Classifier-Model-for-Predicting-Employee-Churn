# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:ROHINI R 
RegisterNumber:  212224240138
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
print(data.head())

data.info()
data.isnull().sum()
data["left"].value_counts()

print(data.head())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])


x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years",
         "salary"]]
print(x.head())


y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(6,8))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)

plt.show()

*/

```

## Output:
![image](https://github.com/user-attachments/assets/66a3f1e4-5c69-4e54-b20c-0722afce30eb)
![image](https://github.com/user-attachments/assets/c464c2a8-b1ee-401c-ad03-a72912a826b6)

![image](https://github.com/user-attachments/assets/da63221f-bcc0-4005-b8ea-0871b766462f)
![image](https://github.com/user-attachments/assets/c5725ebb-9277-430c-95f0-451a2aa9695f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
