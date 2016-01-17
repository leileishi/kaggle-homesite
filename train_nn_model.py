#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd

import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def main():
    model_name = 'Neural Network'

    parser = argparse.ArgumentParser(usage=model_name)

    parser.add_argument("train_feature", help="Input file of training features and target")
    parser.add_argument("test_feature", help="Input file of test features")
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
    train_y1 = df_train_feature_target.values[:,-1]
    train_y0 = np.zeros_like(train_y1) - train_y1
    train_y = np.stack((train_y0, train_y1), axis=-1)
    test_X = df_test_feature.values

    # Normalise train data and test data
    print('Normalise training data and test data')
    scaler = StandardScaler(copy=False)
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # NN Model specification
    def build_model():
        print('Build neural network')
        model = Sequential()
        model.add(Dense(100, input_dim=len(train_X[0]), init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(50, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(2, init='uniform'))
        model.add(Activation('softmax'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mse', optimizer=adam)

        return model

    # Cross validation
    print('Cross validation')
    nb_folds = 5
    kfolds = KFold(len(train_y), nb_folds)
    scores = []
    f = 0
    for train, valid in kfolds:
        print('---'*20)
        print('Fold', f)
        print('---'*20)
        f += 1
        train_xx = train_X[train]
        valid_xx = train_X[valid]
        train_yy = train_y[train]
        valid_yy = train_y[valid]

        print("Training model...")
        clf = build_model()
        clf.fit(train_xx, train_yy, validation_data=(valid_xx, valid_yy), 
            callbacks=[EarlyStopping(patience=5)], 
            nb_epoch=100, batch_size=1000, show_accuracy=True, verbose=2)
        valid_preds = clf.predict_proba(valid_xx, verbose=0)
        valid_preds = valid_preds[:,1]
        score = roc_auc_score(valid_yy[:,1], valid_preds)
        print("ROC AUC:", score)
        scores += [score, ]
        print(scores)

    # Train the full model
    print('Train the full model')
    clf = build_model()
    clf.fit(train_X, train_y, 
        callbacks=[EarlyStopping(patience=5)], 
        nb_epoch=100, batch_size=1000, show_accuracy=True, verbose=2)

    # Make predictions with the full model
    test_pred = clf.predict(test_X)
    test_pred_prob = clf.predict_proba(test_X)[:,1]

    # Write out the prediction result
    print('Write out the prediction result')
    pd.Series(test_pred_prob  if args.prob else test_pred, name='Prob' if args.prob else 'Pred') \
        .to_csv(args.test_pred, index=False, header=True)

    # Report the result
    print('Report the result')
    print('Best Score: ', str(np.mean(scores)))
    print('Best Parameter: ')
    print('Parameter Scores: ')
    print('Model: ')
    print('Accuracy: ')
    print('F1:       ')
    print('ROC AUC:  ')
    print(args.test_pred + '~~' + str(np.mean(scores)))

if __name__ == '__main__':
    main()