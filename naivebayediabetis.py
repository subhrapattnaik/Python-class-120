import pandas as pd
import plotly.express as px



df = pd.read_csv('./120/diabetes.csv')

print(df.head())

#In the data that we have, we can see that we have glucose, bloodpressure and we know if the given person has diabetes or not.
#-------------------------------------------


#Here, we will use the glucose and the bloodpressure to predict if the person has diabetes or not using Naive Bayes.
from sklearn.model_selection import train_test_split

X = df[["glucose", "bloodpressure"]]
y = df["diabetes"]

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

#Here, we can see that we have an accuracy of approximately an outstanding 94.4%.
#------------------------------------------------------------------------
#-------------------------------------------------------------------




#  Let's see if using logistics regression would have given us the same accuracy?


from sklearn.model_selection import train_test_split

X = df[["glucose", "bloodpressure"]]
y = df["diabetes"]

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



#While the accuracy score for both the datasets was close, with Naive Bayes giving us an accuracy of 94.4% and logistic regression giving us an accuracy of 91.6%, Naive Bayes still performed better.

#The reason for this is that if we look at our features again, we can see that the Glucose and the Blood Pressure had no correlation with each other. They both contributed individually to whether a person would have diabetes or not. This is exactly what Naive Bayes algorithm assumes, that all the features contribute individually to the outcome.

#This was for the case of where Naive Bayes outperforms Logistic Regression, but let's see an example of the case where Logistic Regression outperforms Naive Bayes.




