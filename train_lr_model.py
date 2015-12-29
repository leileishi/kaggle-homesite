#!/usr/bin/env python

import sys
import argparse
import pandas as pd

import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser(usage='Logistic Regression')

    parser.add_argument("train_feature", help="Input file of training features and target")
    parser.add_argument("test_feature", help="Input file of test features")
    parser.add_argument("train_pred", help="Output file of predicted training target")
    parser.add_argument("test_pred", help="Output file of predicted test target")
    parser.add_argument("--prob", action='store_true', help='Predict probability of class 1')
    parser.add_argument("--cores", type=int, default=-1, help='Number of cores to use')

    args = parser.parse_args()

    df_train_feature_target = pd.read_csv(args.train_feature)
    df_test_feature = pd.read_csv(args.test_feature)

    train_X = df_train_feature_target.iloc[:,:-1]
    train_y = df_train_feature_target.iloc[:,-1]
    test_X = df_test_feature

    parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    model = LogisticRegression(penalty='l2')

    clf = GridSearchCV(model, parameters, cv=5, scoring='f1', n_jobs=args.cores)
    clf.fit(train_X, train_y)

    print(clf)

    train_pred = clf.predict_proba(train_X)[:,1] if args.prob else clf.predict(train_X)[:]
    test_pred = clf.predict_proba(test_X)[:,1] if args.prob else clf.predict(test_X)

    pd.Series(train_pred, name='Prob').to_csv(args.train_pred, index=False, header=True)
    pd.Series(test_pred, name='Prob').to_csv(args.test_pred, index=False, header=True)

if __name__ == '__main__':
    main()