# Artificial Neural Network

# 1 - Data Preprocessing
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = dataset['Exited']

# Encode categorical data

X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
# Fit/transform to training first so test is based on the same thing
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# 2 - Making the ANN

# Initializing the ANN
classifier = Sequential()

# Adding input layer and first hidden layer
# Units: hidden nodes -> inputs + outputs divided by 2
classifier.add(
    Dense(units=6,
          kernel_initializer='uniform',
          activation='relu',
          input_dim=11))

# Adding the second hidden layer
classifier.add(
    Dense(units=6,
          kernel_initializer='uniform',
          activation='relu'))

# Adding the output layer
# If our dependent variable had >2 categories, we would use softmax over sigmoid
classifier.add(
    Dense(units=1,
          kernel_initializer='uniform',
          activation='sigmoid'))

# Compiling the ANN
# 'adam' is an algorithm for stochastic gradient descent
# If output isn't binary, loss='categorical_crossentropy'
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100)

# 3 - Making predictions and evaluating the model
# Predicting the test set results
y_pred = classifier.predict(X_test)

# Predicting gives probabilities, need to convert them to binary predictions
# This means choosing a threshold
y_pred = (y_pred > 0.5)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Test model on new example
example = np.array([[600, 40, 3, 60000, 2, 1, 1, 50000, 0, 0, 1]])
scaled_example = sc_X.transform(example)
probability = classifier.predict(scaled_example)
new_prediction = probability > 0.5

# 4 - Evaluating and Tuning the ANN
# K-fold is part of sklearn, not Keras -> need to combine
# Use a Keras wrapper that wraps this function into the Keras model

def build_classifier():
    ann_classifier = Sequential()
# Input layer and first hidden layer with dropout
    ann_classifier.add(
        Dense(units=6,
              kernel_initializer='uniform',
              activation='relu',
              input_dim=11))
# Best to start with 10%. Go higher if still overfitting
    ann_classifier.add(Dropout(rate=0.1))
# Second hidden layer with dropout
    ann_classifier.add(
        Dense(units=6,
              kernel_initializer='uniform',
              activation='relu'))
    ann_classifier.add(Dropout(rate=0.1))
# Output layer
    ann_classifier.add(
        Dense(units=1,
              kernel_initializer='uniform',
              activation='sigmoid'))
    ann_classifier.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return ann_classifier


classifier = KerasClassifier(build_fn=build_classifier,
                             batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train,
                             y=y_train, cv=10, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()

def build_classifier_dropout(optimizer):
    ann_classifier = Sequential()
# Input layer and first hidden layer with dropout
    ann_classifier.add(
        Dense(units=6,
              kernel_initializer='uniform',
              activation='relu',
              input_dim=11))
    ann_classifier.add(Dropout(rate=0.1))
# Second hidden layer with dropout
    ann_classifier.add(
        Dense(units=6,
              kernel_initializer='uniform',
              activation='relu'))
    ann_classifier.add(Dropout(rate=0.1))
# Output layer
    ann_classifier.add(
        Dense(units=1,
              kernel_initializer='uniform',
              activation='sigmoid'))
    ann_classifier.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return ann_classifier


# We're tuning batch size and number of epochs so don't specify them here
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters,
                           scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
