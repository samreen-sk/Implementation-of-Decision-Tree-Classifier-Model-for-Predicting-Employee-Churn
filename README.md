# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data – Import the employee dataset with relevant features and churn labels.

2. Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3. Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4. Train Model – Fit the model on the training data.

5. Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6. Visualize & Interpret – Visualize the tree and identify key features influencing churn.

## Program and Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SHAIK SAMREEEN
RegisterNumber:  212223110047
*/
```

```python
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
```
![437687390-000ea6d4-1565-429c-930a-46a07b07d973](https://github.com/user-attachments/assets/168fb555-0343-4a45-99bf-bbc994bb808c)

```
data.info()
data.isnull().sum()
data['left'].value_counts()
```
![437687423-c2e61628-d676-4f2e-ae2c-6d84a1649751](https://github.com/user-attachments/assets/3b633906-73a4-41f8-8d8e-ba0c99617ecb)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![437687483-73cadc61-d913-4794-9250-2e473fac7796](https://github.com/user-attachments/assets/95003466-f2eb-4ae1-8246-38bd15e50c45)

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
![437687482-01345a9c-b22d-4163-a44b-cb6e273cc943](https://github.com/user-attachments/assets/8ade3d28-4b55-4a92-8cf8-224d98aef905)

```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![437687516-df602824-d905-4a06-86cd-4387474ba2c6](https://github.com/user-attachments/assets/cca0cea3-0794-42fd-be21-a3a334aa1f28)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![437687531-bc8439d0-5959-4d52-a5cc-3512b9ff22ec](https://github.com/user-attachments/assets/5c4890cd-4add-479d-b9a3-97c50843407f)
```
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![437687552-69ab36f0-289a-4765-8a63-20f0c165c070](https://github.com/user-attachments/assets/b1b91647-1280-4481-a9af-3c55625cf6f7)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
