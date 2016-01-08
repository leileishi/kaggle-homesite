#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy import stats
import pandas as pd

def weighted_gmean(d, w):
    return np.exp(np.sum(np.log(d)*w) / np.sum(w))

def main():
    parser = argparse.ArgumentParser(usage='Average ranks of model outputs')

    parser.add_argument("model_list", help="Input file of models and weights")
    parser.add_argument("test_pred", help="Output file of predictions")

    args = parser.parse_args()

    print('Average ranks of model outputs')

    model_list = pd.read_csv(args.model_list)
    models = map(lambda m: pd.read_csv(m), model_list.Model)
    ranks = map(lambda m: m.values.T[0].argsort().argsort(), models)
    mean_ranks = map(lambda r: weighted_gmean(r, model_list.Weight.values), zip(ranks))
    norm_ranks = mean_ranks/max(mean_ranks)

    pd.Series(norm_ranks, name='Prob').to_csv(args.test_pred, index=False, header=True)

if __name__ == '__main__':
    main()