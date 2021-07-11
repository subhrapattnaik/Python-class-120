import pandas as pd
import plotly.express as px



df = pd.read_csv('./120/income.csv')

print(df.head())
print(df.describe())

#From the given data, we will consider the following fields to determine the salary of a person -

#Age
#Hours Per Week
#Education Number
#Capital Gain
#Capital Loss
#-------------------------------------------


#Here, we will use the glucose and the bloodpressure to predict if the person has diabetes or not using Naive Bayes.
from sklearn.model_selection import train_test_split


X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.25, random_state=42)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.25, random_state=42)
#Training the model with naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()

x_train_1 = sc.fit_transform(x_train_1) 
x_test_1 = sc.fit_transform(x_test_1) 

model_1 = GaussianNB()
model_1.fit(x_train_1, y_train_1)

y_pred_1 = model_1.predict(x_test_1)

accuracy = accuracy_score(y_test_1, y_pred_1)
print(accuracy)

#This time, with the new dataset, we can see that Naive Bayes gave us an accuracy of almost 79%. 
#------------------------------------------------------------------------
#-------------------------------------------------------------------




#  Let's see how much accuracy do we get with Logistic Regression.


from sklearn.model_selection import train_test_split

X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.25, random_state=42)


#Training the model with Logistic Regression

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()

x_train_2 = sc.fit_transform(x_train_2) 
x_test_2 = sc.fit_transform(x_test_2) 

model_2 = LogisticRegression(random_state = 0) 
model_2.fit(x_train_2, y_train_2)

y_pred_2 = model_2.predict(x_test_2)

accuracy = accuracy_score(y_test_2, y_pred_2)
print(accuracy)



#With Logistic Regression, this time, we got an accuracy of 81.1%. Let's study this more closely..

#Difference b/w Naive Bayes and Logistic Regression
#In the first dataset, as we pointed out earlier, both the glucose and the bloodpressure had little correlation, and both of them were contributing individually to whether a person has diabetes or not.

#Conclusion In these kinds of dataset, where all the features contribute individually to the outcome, Naive Bayes outperforms logistic regression and is highly efficient.

#In the second dataset, Logistic Regression outperformed Naive Bayes. The reason is that in this dataset, not all features contribute individually to the outcome. For example, there have been people of all age groups earning both less than and more than 50K. There have also been people with all education numbers that have an income of both less and more than 50K. Here, the combination of all the features is a better predictor of whether a person is earning more than or less than 50K, instead of all features having their individual contribution.



