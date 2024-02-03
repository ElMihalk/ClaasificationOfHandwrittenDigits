import pandas as pd
import tensorflow as tf
import numpy as np
import keras

#Stage 1/5: The Keras dataset
(X_initial, Y_initial), (_, _) = keras.datasets.mnist.load_data()
X_initial = X_initial.reshape((X_initial.shape[0], X_initial.shape[1]*X_initial.shape[2]))
# print(f'Classes: {np.unique(Y_initial)}')
# print(f'Features\' shape: {X_initial.shape}')
# print(f'Target\'s shape: {Y_initial.shape}')
# print(f'min: {np.min(X_initial)}, max: {np.max(X_initial)}')

#Stage 2/5: Split into sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_initial[:6000], Y_initial[:6000], train_size=0.7, random_state=40)

# print(f'x_train shape: {X_train.shape}')
# print(f'x_test shape: {X_test.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(f'y_test shape: {y_test.shape}')
# print('Proportion of samples per class in train set:')
# print(pd.Series(y_train).value_counts(normalize=True))

#Stage 3/5: Train models with default settings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score

# the function
def fit_predict_eval(model, features_train, features_test, target_train, target_test, display=False):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(y_true=target_test, y_pred=y_pred)
    if display:
        print(f'Model: {model}\nAccuracy: {score}\n')
    return score

model_dict = {
    'neighbor': KNeighborsClassifier(),
    'tree': DecisionTreeClassifier(random_state=40),
    'log_reg': LogisticRegression(random_state=40),
    'forest': RandomForestClassifier(random_state=40),
              }

score_board = {}
for algo in model_dict.keys():
    score = fit_predict_eval(
            model=model_dict[algo],
            features_train=X_train,
            features_test=X_test,
            target_train=y_train,
            target_test=y_test
        )
    score_board[score] = model_dict[algo]
max_accuracy = max(score_board.keys())

# print(f"The answer to the question: {str(score_board[max_accuracy]).split('(')[0]} - {max_accuracy}")

#Stage 4/5: Data preprocessing
from sklearn.preprocessing import Normalizer

def norm_check(score_initial={}, score_norm={}):
    comparison = [1 for i, j in zip(score_initial.keys(), score_norm.keys())]
    if sum(comparison) > len(score_initial.keys())/2:
        return 'yes'
    else:
        return 'no'

#Initialize the normalizer
normalizer = Normalizer()

#Transform the features
X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)

score_board_norm = {}
for algo in model_dict.keys():
    score = fit_predict_eval(
            model=model_dict[algo],
            features_train=X_train_norm,
            features_test=X_test_norm,
            target_train=y_train,
            target_test=y_test,
            display=False
        )
    score_board_norm[score] = model_dict[algo]
max_accuracy_norm = max(score_board_norm.keys())

#Does the normalization have a positive impact in general? (yes/no)
# print(f"The answer to the 1st question: {norm_check(score_board, score_board_norm)}")
# print()

#Which two models show the best scores?
first_score_norm = score_board_norm[max_accuracy_norm]
del score_board_norm[max_accuracy_norm]
second_accuracy_norm = max(score_board_norm.keys())
second_score_norm = score_board_norm[second_accuracy_norm]
# print(f"The answer to the 2nd question: {str(first_score_norm).split('(')[0]} - {max_accuracy}, {str(second_score_norm).split('(')[0]} - {second_accuracy_norm}")

#Stage 5/5: Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
#KNeighbor and RandomForest | X normalized

#KNeighborClassifier grid search

neighbor_param_grid = {
    'n_neighbors': [3, 4],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute']
}

neighbor_grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=neighbor_param_grid,
    n_jobs=-1
)

neighbor_grid_search.fit(X_train_norm, y_train)
neighbor_y_pred = neighbor_grid_search.predict(X_test_norm)
neighbor_accuracy = accuracy_score(y_test, neighbor_y_pred)

#RandomForest grid search

forest_param_grid = {
    'n_estimators': [300, 500],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
}

forest_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=40),
    param_grid=forest_param_grid,
    n_jobs=-1
)

forest_grid_search.fit(X_train_norm, y_train)
forest_y_pred = forest_grid_search.predict(X_test_norm)
forest_accuracy = accuracy_score(y_test, forest_y_pred)

print('K-nearest neighbours algorithm')
print(f'best estimator: {neighbor_grid_search.best_estimator_}')
print(f'accuracy: {neighbor_accuracy}')
print()

print('Random forest algorithm')
print(f'best estimator: {forest_grid_search.best_estimator_}')
print(f'accuracy: {forest_accuracy}')



