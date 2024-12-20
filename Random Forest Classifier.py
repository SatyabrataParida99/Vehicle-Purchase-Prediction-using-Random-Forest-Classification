# Random Forest Classification

# # Importing the libraries
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For data visualization
import pandas as pd # For data manipulation and analysis

# Importing the dataset
data = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Vehicle purchase Prediction.csv")
x = data.iloc[:, [2, 3]].values # Extracting 'Age' and 'Estimated Salary' as independent variables
y = data.iloc[:, -1].values  # Extracting the target variable (Purchased)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  # For splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)  # 80% training, 20% testing split

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler  # For standardizing the data
sc = StandardScaler()  # Initializing the scaler
x_train = sc.fit_transform(x_train)  # Scaling the training data
x_test = sc.transform(x_test)  # Scaling the test data
'''

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier   # Importing the Random Forest Classifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=70, random_state=0, max_depth=6)  # Creating the classifier with specific parameters
classifier.fit(x_train, y_train)  # Training the classifier on the training data

# Predicting the Test set results
y_pred = classifier.predict(x_test)  # Making predictions on the test data

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix  # For evaluating the classification model
cm = confusion_matrix(y_test, y_pred)  # Generating the confusion matrix
print(cm)  # Printing the confusion matrix

# Calculating the accuracy score
from sklearn.metrics import accuracy_score  # For calculating accuracy
ac = accuracy_score(y_test, y_pred)  # Computing accuracy score
print(ac)  # Printing the accuracy


# Calculating training accuracy (bias)
bias = classifier.score(x_train, y_train)  # Evaluating model accuracy on training data
bias  # Printing training accuracy

# Calculating test accuracy (variance)
variance = classifier.score(x_test, y_test)  # Evaluating model accuracy on test data
variance  # Printing test accuracy

# Visualising the Training set results
from matplotlib.colors import ListedColormap  # For creating custom colormaps
x_set, y_set = x_train, y_train  # Assigning training data for visualization
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),  # Creating grid for X1
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))  # Creating grid for X2
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))  # Plotting decision boundary
plt.xlim(x1.min(), x1.max())  # Setting limits for X1
plt.ylim(x2.min(), x2.max())  # Setting limits for X2
for i, j in enumerate(np.unique(y_set)):  # Looping through unique labels
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)  # Plotting data points for each class
plt.title('Random Forest Classification (Training set)')  # Adding plot title
plt.xlabel('Age')  # Labeling x-axis
plt.ylabel('Estimated Salary')  # Labeling y-axis
plt.legend()  # Adding legend
plt.show()  # Displaying the plot

# Visualising the Test set results
x_set, y_set = x_test, y_test  # Assigning test data for visualization
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),  # Creating grid for X1
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))  # Creating grid for X2
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))  # Plotting decision boundary
plt.xlim(x1.min(), x1.max())  # Setting limits for X1
plt.ylim(x2.min(), x2.max())  # Setting limits for X2
for i, j in enumerate(np.unique(y_set)):  # Looping through unique labels
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)  # Plotting data points for each class
plt.title('Random Forest Classification (Test set)')  # Adding plot title
plt.xlabel('Age')  # Labeling x-axis
plt.ylabel('Estimated Salary')  # Labeling y-axis
plt.legend()  # Adding legend
plt.show()  # Displaying the plot