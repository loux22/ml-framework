import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

# ========== DECISION TREE CLASSIFIER ==========
from ml.tree import DecisionTreeClassifier
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("ml/tree/iris.csv", skiprows=1, header=None, names=col_names)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)


classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test) 
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy : {}\nPercentage accuracy : {}%'.format(accuracy, round(accuracy * 100,2)))

# ========== LINEAR REGRESSION ==========
from ml.regression.linear import LinearRegression
df = pd.read_csv("data/salary_data.csv")

X = df[["YearsExperience"]].values
y = df[["Salary"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

Y_pred = model.predict(X_test)

plt.scatter( X_test, y_test, color = 'blue' )
plt.plot( X_test, Y_pred, color = 'orange' )
plt.title( 'Salary vs Experience' )
plt.xlabel( 'Years of Experience' )
plt.ylabel( 'Salary' )
plt.show()




# from ml.regression.linear import LinearRegression

# X, y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=6, bias=30)
# m = 200

# linear_reg = LinearRegression(X)

# plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
# linear_reg.graph(linear_reg.my_formula, range(-5,5))
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()