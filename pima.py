import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import scikit libs
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB

# import performance metrics library
from sklearn import metrics

# Logistic Regression
from sklearn.linear_model import LogisticRegression


def plot_correlation(df, size=11):
    """
    Function plots a graphic correlation matrix for each pari of columns in the dataframe.
    :param df: Pandas dataframe
    :param size: vertilca nad horisontal size of the plot
    Displays:
    Matrix of correlation between columns. Blue-cyan-yellow-red-darkred => less to more correlated
    """
    corr = df.corr()  # Data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # Color code the rectanges by correlation
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

# Load review data
df = pd.read_csv("pima-data.csv")


del df['skin']

diabetes_map = {True: 1, False: 0}

df['diabetes'] = df['diabetes'].map(diabetes_map)

num_true = len(df.loc[df['diabetes']==True])
num_false = len(df.loc[df['diabetes']==False])

print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true+num_false))*100))
print("Number of True cases: {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true+num_false))*100))

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin',  'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

x = df[feature_col_names].values
y = df[predicted_class_names].values

split_test_size = 0.30

# test_size = 0.3 is 30%, 42 is the answer to everything(randomization static number)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)


# Verify predicted values was split correct
### Post-split Data preparation
print('Glucose_conc zeros: ', len(df.loc[df['glucose_conc']==0]))
print('Diastolic zeros: ', len(df.loc[df['diastolic_bp']==0]))
print('thickness zeros: ', len(df.loc[df['thickness']==0]))
print('insulin zeros: ', len(df.loc[df['insulin']==0]))
print('bmi zeros: ', len(df.loc[df['bmi']==0]))
print('diab_pred zeros: ', len(df.loc[df['diab_pred']==0]))
print('age zeros: ', len(df.loc[df['age']==0]))


# Fill all "0"s with mean of the column
fill_0 = Imputer(missing_values=0, strategy='mean', axis=0)

x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)



# Train the model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train.ravel())

GaussianNB()

# Check the performance on training data
nb_predict_train = nb_model.predict(x_train)

print('Accuracy: {0:4f}'.format(metrics.accuracy_score(y_train, nb_predict_train)))

nb_predict_test = nb_model.predict(x_test)
# predict on test data
print('Accuracy test: {0:.4f}'.format(metrics.accuracy_score(y_test, nb_predict_test)))

# Metrics
print('Confusion metrics')
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1, 0]))
print('Classification report')
print(metrics.classification_report(y_test, nb_predict_test, labels=[1, 0]))

#
# Random Forest Model
#
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train.ravel())

rf_predict_train = rf_model.predict(x_train)

print('Accuracy: {0:4f}'.format(metrics.accuracy_score(y_train, rf_predict_train)))

rf_predict_test = rf_model.predict(x_test)

# Training metrics
print('Accuracy: {0:.4f}'.format(metrics.accuracy_score(y_test, rf_predict_test)))

# Metrics
print('Confusion metrics')
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0]))
print('Classification report')
print(metrics.classification_report(y_test, rf_predict_test, labels=[1, 0]))


# Logistic Regression
lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(x_train, y_train.ravel())

lr_predict_test = lr_model.predict(x_test)

print('=======Logistic Regression=======')
# Metrics
print('Confusion metrics')
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1, 0]))
print('Classification report')
print(metrics.classification_report(y_test, lr_predict_test, labels=[1, 0]))


# Setting regularization parameters
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0

while(C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight='balanced', random_state=42)
    lr_model_loop.fit(x_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(x_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val += C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print('1st max value of {0:.3f} occured at C={1:.3f}'.format(best_recall_score, best_score_C_val))


print('============')

print('C_value: ', C_values)
print('Recal Score: ', recall_scores)

print('============')


plt.plot(C_values, recall_scores, '-')
plt.xlabel('C value')
plt.ylabel('recal score')

plt.show()

from sklearn.linear_model import LogisticRegression

lr_model=LogisticRegression(class_weight='balanced', C=best_score_C_val, random_state=42)

print('=======Logistic Regression Final=======')

# print('Accuracy: {0:4f}'.format(metrics.accuracy_score(y_train, nb_predict_train)))
#
# nb_predict_test = nb_model.predict(x_test)
# # predict on test data
# print('Accuracy test: {0:.4f}'.format(metrics.accuracy_score(y_test, nb_predict_test)))
#
# # Metrics
# print('Confusion metrics')
# # Note the use of labels for set 1=True to upper left and 0=False to lower right
#
# print(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1, 0]))
# print('Classification report')
# print(metrics.classification_report(y_test, nb_predict_test, labels=[1, 0]))