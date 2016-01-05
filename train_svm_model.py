#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.svm import SVC

def main():
    model_name = 'Support Vector Machine'

    parser = argparse.ArgumentParser(usage=model_name)

    parser.add_argument("train_feature", help="Input file of training features and target")
    parser.add_argument("test_feature", help="Input file of test features")
    parser.add_argument("train_pred", help="Output file of predicted training target")
    parser.add_argument("test_pred", help="Output file of predicted test target")
    parser.add_argument("--prob", action='store_true', help='Predict probability of class 1')
    parser.add_argument("--cores", type=int, default=-1, help='Number of cores to use')

    args = parser.parse_args()

    print(model_name)

    # Read training data and test data
    print('Read training data and test data')
    df_train_feature_target = pd.read_csv(args.train_feature, dtype=np.float32)
    df_test_feature = pd.read_csv(args.test_feature, dtype=np.float32)

    train_X = df_train_feature_target.values[:,:-1]
    train_y = df_train_feature_target.values[:,-1]
    test_X = df_test_feature.values

    # Normalise training data and test data
    scaler = StandardScaler(copy=False)
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Model specification and parameter range
    model = SVC(probability=True)
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Cross validation search
    print('Cross validation search')
    clf = GridSearchCV(model, parameters, 
        cv=5, scoring='roc_auc', n_jobs=args.cores, pre_dispatch=args.cores, verbose=3)
    clf.fit(train_X, train_y)

    # Make predictions with the best model
    print('Make predictions with the best model')
    train_pred = clf.predict(train_X)
    train_pred_prob = clf.predict_proba(train_X)[:,1]
    test_pred = clf.predict(test_X)
    test_pred_prob = clf.predict_proba(test_X)[:,1]

    # Write out the prediction result
    print('Write out the prediction result')
    pd.Series(train_pred_prob if args.prob else train_pred , name='Prob' if args.prob else 'Pred') \
        .to_csv(args.train_pred, index=False, header=True)
    pd.Series(test_pred_prob  if args.prob else test_pred, name='Prob' if args.prob else 'Pred') \
        .to_csv(args.test_pred, index=False, header=True)

    # Report the result
    print('Report the result')
    print('Accuracy: ', accuracy_score(train_y, train_pred))
    print('F1:       ', f1_score(train_y, train_pred))
    print('ROC AUC:  ', roc_auc_score(train_y, train_pred_prob))
    print('Model: ', clf)
    print('Best Parameter: ', clf.best_params_)
    print('Best Score: ', clf.best_score_)
    print('Parameter Scores: ', clf.grid_scores_)

if __name__ == '__main__':
    main()