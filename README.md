# HR_logistic_regression

Logistic regression is a simple machine learning method that we can use to predict the value of a numeric categorical variable based on its relationship with predictor variables. Logistic regression differs with linear regression. Logistic regression is used to <strong> predict categories for ordinal variables </strong>, whereas, linear regression is used to predict values for numeric continuous variables.


<strong><h2> Employees Attrition/Churn Analysis using Logistic Regression </strong></h2>

Employees are the valuable asset in companies/organizations. Without employees, companies/organizations could not meet their goals and grow. Employees attrition is a serious issue that needs to pay attention by companies/organizations to solve. Therefore, HR data is vital to identify which areas that need to be modified/improved to ensure the employees to stay for a longer period.


<h3> 1) Import data analysis library and read dataset </h3>

```bash
import pandas as pd

HR_dataset = pd.read_csv('https://raw.githubusercontent.com/theleadio/datascience_demo/master/HR_dataset.csv')
```


<h3> 2) Overview the head of dataset </h3>

```bash
HR_dataset.head()
```


<h3> 3) Overview dimension of dataset </h3>

```bash
HR_dataset.shape
```


<h3> 4) Overview number of employees attrition </h3>

```bash
HR_dataset['left'].value_counts()
```


<h3> 5) Import visualization libraries and visualize the employees resignation </h3>

```bash
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = 'left', data = HR_dataset, palette = "hls")
plt.show()
```


<h3> 6) Define variables </h3>
Define variables/parameters that we want to use to build logistic regression model based on training set.
Define x- and y-axis

```bash
parameters = ['satisfaction_level', 'exp_in_company', 'average_monthly_hours']

x = HR_dataset[parameters]
y = HR_dataset['left']
```


<h3> 7) Import machine learning library and split the dataset </h3>

```bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
```


<h3> 8) Build and fit the logistic regression model using training set </h3>

```bash
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
```


<h3> 9) Evaluate the accuracy of logistic regression model using testing set </h3>

```bash
y_prediction = log_model.predict(x_test)

print("Accuracy", (log_model.score(x_test, y_test)))
```


<h3> 10) Build the confusion matrix using testing set </h3>

```bash
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_prediction)
print(confusion_matrix)
```


<h3> Conclusion </h3>




<h3> Reference: </h3>
The Python codes were adapted and slightly modified based on Besant Technologies' Youtube https://youtu.be/9ZiPeNVFKgI and Dr Cher Han Lau 
