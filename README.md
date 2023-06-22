# Python-class-120


https://towardsdatascience.com/how-to-drop-rows-in-pandas-dataframes-with-nan-values-in-certain-columns-7613ad1a7f25


# Apply logistic regression

df = df.dropna(subset=["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"])
x = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df['income']
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x, y, test_size=0.25, random_state=42)
sc = StandardScaler()
x_train_2 = sc.fit_transform(x_train_2)
x_test_2 = sc.fit_transform(x_test_2)
model_2 = LogisticRegression(random_state = 0)
model_2.fit(x_train_2, y_train_2)
y_pred_2 = model_2.predict(x_test_2)
accuracy = accuracy_score(y_test_2, y_pred_2)
print(accuracy)
