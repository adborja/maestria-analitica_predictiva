# Wine quality

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (red wine)
dataset = pd.read_csv('winequality-red.csv', sep = ';')
random_state = 123

# Shuffle the dataframe
from sklearn.utils import shuffle
dataset = shuffle(dataset, random_state = random_state)
dataset = shuffle(dataset)
dataset = shuffle(dataset)

X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

# Show the dimensions of dataset
print(dataset.shape)

# Describe the dataset
print(dataset.describe())

# Print class (quality) distribution
print(dataset.groupby('quality').size())

# Histogram (Univariate plot)
dataset.hist()

# Multivariate plot
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = random_state)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Evaluating models (Applying k-fold cross validation)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RDF', RandomForestClassifier()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    # kfold = RepeatedKFold(n_splits = 10, n_repeats = 2, random_state = 0)
    kfold = StratifiedKFold(n_splits = 3, shuffle = False, random_state = random_state)
    cv_results = cross_val_score(estimator = model, X = X_train, y = y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comparing algorithms
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Fitting SVC classifier to the Training set
from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = random_state)
classifier = SVC(C = 100, gamma = 1.0, class_weight = 'balanced', random_state = random_state)
classifier.fit(X_train, y_train)

# Fitting Random Forest classifier to the Training set (best classifier)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = random_state)
classifier.fit(X_train, y_train)

# Plotting the feature importances
importances = classifier.feature_importances_
names = dataset.columns
importances, names = zip(*sorted(zip(importances, names)))
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis = 0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.barh(range(len(names)), importances, align = 'center', color = 'r', xerr = std[indices])
plt.yticks(range(len(names)), names)
plt.ylabel('Features')
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
cm_df = pd.DataFrame(cm,
                     index = ['3', '4', '5', '6', '7', '8'],
                     columns = ['3', '4', '5', '6', '7', '8'])

plt.figure(figsize = (5.5, 4))
sns.heatmap(cm_df, annot = True)
plt.title('Random Forest Classifier \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Applying grid search to find the best model and the best parameters (SVC)
from sklearn.model_selection import GridSearchCV

# SVC
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]}]

# Random Forest Classifier
parameters = [{'criterion': ['gini'], 'n_estimators': [10, 15, 20, 50, 100, 200, 300, 400, 500]},
              {'criterion': ['entropy'], 'n_estimators': [10, 15, 20, 50, 100, 200, 300, 400, 500], 'max_features': ['auto', 'sqrt', 'log2', None]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
