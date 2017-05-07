import csv
import os 
from math import sqrt
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
from get_data import get_data
from get_sample import read_samples

filename = "data.csv"
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", filename)
EPSILON = 0.0001

def get_ratings(samples=None):
    return get_data(format="rating_list",samples=samples)

def rmse(ratings, samples):
    ret = 0.0
    for sample in samples:
        ret += (sample[2] - ratings[sample[0],sample[1]]) ** 2
    ret /= len(samples)
    ret = sqrt(ret)
    return ret

# Given: List of ratings representing an MxN sparse matrix
# Returns: MxN matrix of rating approximations
def sgd(ratings, user_indexes, course_indexes, max_k=25, alpha=0.01, regularizer=0.02, default=0.1, max_iterations=100):
    U = np.full((max_k,len(user_indexes)), default)
    V = np.full((max_k,len(course_indexes)), default)

    for feature in np.arange(max_k):
        prev_norm_U = float('Inf')
        prev_norm_V = float('Inf')
        for iteration in np.arange(max_iterations):
            print("k: %d, iteration: %d", (feature, iteration))
            for rating in ratings:
                user_feature = U[feature,rating.user_id]
                course_feature = V[feature,rating.course_id]
                rating.error = rating.value - user_feature*course_feature
                U[feature,rating.user_id] += alpha * (rating.error*course_feature - regularizer*user_feature)
                V[feature,rating.course_id] += alpha * (rating.error*user_feature - regularizer*course_feature)

            norm_U = la.norm(U)
            norm_V = la.norm(V)
            if abs(norm_U - prev_norm_U) < EPSILON and abs(norm_V - prev_norm_V) < EPSILON:
                print("Feature has converged.")
                break
            prev_norm_U = norm_U
            prev_norm_V = norm_V

        for rating in ratings:
            rating.value = rating.error
    return np.transpose(np.transpose(U).dot(V))

if __name__ == '__main__':
    samples = read_samples()
    ratings, user_indexes, course_indexes = get_ratings(samples)
    # print(sgd(ratings, user_indexes, course_indexes))
    sgd_ratings = sgd(ratings, user_indexes, course_indexes,max_k=10, max_iterations=30)
    # sgd_ratings = sgd(ratings, user_indexes, course_indexes)
    print(rmse(sgd_ratings,samples))