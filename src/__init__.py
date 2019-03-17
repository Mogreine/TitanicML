import matplotlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from xgboost import XGBClassifier
import functools


def norm_data(df):
    names = list(df)
    scaler = preprocessing.MinMaxScaler()
    scaled_columns = scaler.fit_transform(df.values)
    res = pd.DataFrame(scaled_columns, columns=names)
    return res


def process_data(data):
    new_data = data
    new_data['Fare'] = new_data['Fare'].fillna(new_data['Fare'].median())
    new_data['Age'] = new_data['Age'].fillna(new_data['Age'].mean())
    # Cast the Age column to the int
    new_data['Age'] = new_data['Age'].astype(int)
    new_data['Age'] = new_data['Age'].apply(lambda age: 0 if age < 16 else 1)

    # Filing missing embarked data
    new_data['Embarked'] = new_data['Embarked'].fillna('S')
    new_data['Embarked'] = new_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Making new features based on family size
    new_data['FamilySize'] = new_data['Parch'] + new_data['SibSp'] + 1
    new_data['IsAlone'] = 0
    new_data.loc[new_data['FamilySize'] == 1, 'IsAlone'] = 1

    # Title feature adding
    titles = list(map(lambda name: name.split('. ')[0].split(' ')[-1], new_data['Name']))
    new_data['Title'] = titles
    # unique_titles = set(titles)
    common = ['Mr', 'Miss', 'Mrs', 'Ms', 'Mlle', 'Dr', 'Rev', 'Mme']
    mst = ['Master']
    higher = ['Sir', 'Lady']
    new_data['Title'] = new_data['Title'].apply(lambda title: 0 if title in common else 1 if title in higher else
                                                2 if title in mst else 3).astype(int)

    # Mapping Sex
    new_data['Sex'] = new_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping Age
    # new_data.loc[new_data['Age'] <= 16, 'Age'] = 0
    # new_data.loc[(16 < new_data['Age']) & (new_data['Age'] <= 40), 'Age'] = 1
    # new_data.loc[40 < new_data['Age'], 'Age'] = 2

    # dropping not considerable features
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked']
    if 'Survived' in list(new_data):
        drop_columns += ['Survived']
    new_data = new_data.drop(drop_columns, axis=1)

    nas = [col for col in new_data if new_data[col].hasnans]

    return new_data


def get_log_reg_precision(X_train, X_val, y_train, y_val):
    model = LogisticRegression(random_state=1)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)


def get_random_forest_precision(X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(random_state=1, max_depth=2)
    model.fit(X_train, y_train)
    prec = model.score(X_val, y_val)
    return prec


def test(clf, X_val, y_val):
    return clf.score(X_val, y_val)


def test_xgb_settings(X_train, y_train):
    log = pd.DataFrame(columns=['learning rate', 'n_estimators', 'mean', 'std'])
    rates = [0.02 for i in range(6)]
    for i in range(1, 6):
        rates[i] = rates[i - 1] * 2
    gammas = [0.1 + 0.05 * i for i in range(20)]
    for gam in gammas:
        for i in range(100, 1001, 50):
            model = XGBClassifier(random_state=1, l_rate=0.1, n_est=i, gamma=gam)
            score = test_estimator(model, X_train, y_train)
            log = log.append({'gamma': gam, 'n_estimators': i,
                              'mean': score.mean(), 'std': score.std()}, ignore_index=True)
    return log


def test_estimator(clf, X_val, y_val):
    n_folds = 5
    score = cross_val_score(clf, X_val, y_val, cv=n_folds, n_jobs=-1, scoring='accuracy')
    return score


def write_final_data(clf, test_data, ids):
    test_preds = clf.predict(test_data)
    test_preds = test_preds.reshape((1, test_preds.size))[0]
    output = pd.DataFrame({'PassengerId': ids.values, 'Survived': test_preds})
    output.to_csv('../data/result.csv', columns=['PassengerId', 'Survived'], index=False)


def write_tests(df):
    df.to_excel('../data/xgb_tests2.xlsx', index=False)


def build_nn(optimizer=optimizers.Adam(lr=0.001)):
    ann = Sequential()
    ann.add(Dense(units=7, kernel_initializer='glorot_uniform', activation='relu', input_shape=(7,)))

    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return ann

# [KerasClassifier(build_fn=build_nn, epochs=100, batch_size=16, verbose=0),
#                   RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=3),
#                   LogisticRegression(random_state=1, verbose=0, n_jobs=-1, tol=1e-2),
#                   SVC(random_state=1, kernel='linear', C=0.025),
#                   KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=4)]


class StackModel(object):
    def __init__(self, upper_clf, models, names):
        self.upper_clf = upper_clf
        self.models = models
        self.names = names

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        preds = list()
        for i in range(len(self.models)):
            preds.append(self.models[i].predict(X))
        preds[0] = preds[0].reshape((1, preds[0].size))[0]
        series = [pd.Series(data=pred.tolist()) for pred in preds]
        base_predictions = pd.DataFrame()
        for i in range(len(preds)):
            base_predictions[self.names[i]] = series[i]
        base_predictions = pd.concat([base_predictions, X], axis=1)
        self.upper_clf.fit(base_predictions, y)
        return self

    def predict(self, X_val):
        preds = list()
        for i in range(len(self.models)):
            preds.append(self.models[i].predict(X_val))
        preds[0] = preds[0].reshape((1, preds[0].size))[0]
        series = [pd.Series(data=pred.tolist()) for pred in preds]
        base_predictions = pd.DataFrame()
        for i in range(len(preds)):
            base_predictions[self.names[i]] = series[i]
        base_predictions = pd.concat([base_predictions, X_val], axis=1)
        return self.upper_clf.predict(base_predictions)


def my_cross_val(base_models, names, upper_clf, X, y):
    kf = KFold(n_splits=5, random_state=1)
    scores = list()
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        preds = list()
        upper_train = X_train
        upper_val = X_val
        for i in range(len(base_models)):
            base_models[i].fit(X_train, y_train)
            preds.append(base_models[i].predict(X_train))
            if names[i] == 'Neural network':
                preds[-1] = preds[-1].reshape((1, preds[-1].size))[0]
            upper_train[names[i]] = preds[-1]
            upper_val[names[i]] = base_models[i].predict(X_val)
        upper_clf.fit(upper_train, y_train)
        scores.append(accuracy_score(y_val, upper_clf.predict(X_val)))
    scores = np.array(scores)
    return scores


def main():
    print("Program started.")

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    train_predictors = process_data(train_data)
    train_predictors = norm_data(train_predictors)

    target = train_data['Survived']

    test_predictors = process_data(test_data)
    test_predictors = norm_data(test_predictors)

    # wrapper = KerasClassifier(build_fn=build_nn, epochs=100, batch_size=16, verbose=0)
    # nn_score = test_estimator(wrapper, train_predictors, target)
    # print("Mean accuracy of ann: %f, std is %f" % (nn_score.mean(), nn_score.std()))

    # rf = SVC(random_state=1, kernel='linear', C=0.025)
    # rf.fit(train_predictors, target)
    # rf_score = rf.predict(test_predictors)
    # df = pd.DataFrame({'shit': rf_score})

    # print("Mean accuracy of rf: %f, std is %f" % (rf_score.mean(), rf_score.std()))

    # lr = LogisticRegression(random_state=1, verbose=0, n_jobs=-1, tol=1e-2)
    # lr_score = test_estimator(lr, train_predictors, target)
    # print("Mean accuracy of lr: %f, std is %f" % (lr_score.mean(), lr_score.std()))

    # svc = SVC(random_state=1, kernel='linear', C=0.025)
    # svc_score = test_estimator(svc, train_predictors, target)
    # print("Mean accuracy of lr: %f, std is %f" % (svc_score.mean(), svc_score.std()))

    # nei = KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=4)
    # nei_score = test_estimator(nei, train_predictors, target)
    # print("Mean accuracy of lr: %f, std is %f" % (nei_score.mean(), nei_score.std()))



    # X_train, X_val, y_train, y_val = train_test_split(train_predictors, target, test_size=0.2)
    # super_mod = StackModel()
    # super_mod.fit(X_train, y_train)
    # preds = super_mod.predict(X_val)
    # score = accuracy_score(y_val, preds)
    # print("Super model accuracy: %f" % score)

    # xgb = XGBClassifier(random_state=1, n_jobs=-1, n_estimators=150)
    # models = [KerasClassifier(build_fn=build_nn, epochs=100, batch_size=16, verbose=0),
    #           RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=3)]
    # names = ['Neural network', 'Random forest']
    # # score = my_cross_val(models, names, xgb, train_predictors, target)
    # print(f"Cross val mean score: {score.mean()}, std: {score.std()}")

    # xgb = XGBClassifier(random_state=1, n_jobs=-1).fit(train_predictors, target)
    # imp = xgb.feature_importances_
    # print(list(train_predictors))
    # print(imp)
    # clf = KerasClassifier(build_fn=build_nn, epochs=1000, batch_size=1024, verbose=0)
    clf = XGBClassifier(random_state=1, n_estimators=100, n_jobs=-1)
    nn_score = test_estimator(clf, train_predictors, target)
    print("mean: %f, std: %f" % (nn_score.mean(), nn_score.std()))

    # Making output
    # clf = XGBClassifier(random_state=1, n_jobs=-1)
    clf.fit(train_predictors, target)
    write_final_data(clf, test_predictors, test_data['PassengerId'])

    print("Done!")


if __name__ == '__main__':
    main()
