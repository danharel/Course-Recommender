import numpy as np
import json

from get_data import get_data

N = 30
FILENAME = '../data/samples.txt'

def write_samples():
    X = get_data(format="COO")
    # samples = np.random.choice(X.tolist(), N, False)
    samples = ref_points = [[X.row[i],X.col[i],X.data[i]] for i in np.random.choice(range(len(X.data)),size=N,replace=False)]
    samples = np.array(samples)
    np.savetxt(FILENAME, samples, ['%d', '%d', '%.18e'])

def read_samples():
    return np.loadtxt(FILENAME, dtype=[('column',int),('row',int),('rating',float)])

if __name__ == '__main__':
    write_sample()