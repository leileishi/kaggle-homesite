#!/usr/bin/env python

import sys
import pandas as pd
import argparse

import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser(usage='Prepare the submit file')

    parser.add_argument("test_input", help="Input file of test data")
    parser.add_argument("test_pred", help="Input file of the prediction result")
    parser.add_argument("test_submit", help="Output file for submit")

    args = parser.parse_args()

    print('Prepare the submit file')

    df_test_data = pd.read_csv(args.test_input)
    df_test_pred = pd.read_csv(args.test_pred)

    if len(df_test_data.index) != len(df_test_pred.index):
        raise Exception('The test prediction doesn\'t match test data')

    id_col = df_test_data.columns[0]
    target_col = 'QuoteConversion_Flag'

    df_test_submit = pd.concat([df_test_data[id_col], df_test_pred], axis=1)
    df_test_submit.columns = [id_col, target_col]
    df_test_submit.to_csv(args.test_submit, index=False, header=True)

if __name__ == '__main__':
    main()