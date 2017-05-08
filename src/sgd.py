import csv
import os 
from math import sqrt
from itertools import chain
from pdb import set_trace

import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse

from get_data import get_data
from get_sample import read_samples, read_user_sample
from compute_predictions import maximization_iteration_zeroes

filename = "data.csv"
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", filename)
EPSILON = 0.0001

def get_ratings(grade_samples=None,user_samples=None):
    return get_data(format="rating_list",grade_samples=grade_samples,user_samples=user_samples)

def rmse(ratings, samples):
    ret = 0.0
    for sample in samples:
        ret += (sample[2] - ratings[sample[0],sample[1]]) ** 2
    ret /= len(samples)
    ret = sqrt(ret)
    return ret

def rank_difference(rank, rank_range):
    if rank < rank_range[0]:
        return rank_range[0] - rank
    if rank >= rank_range[1]:
        return rank - rank_range[1] + 1
    else:
        return 0

def ranking_error(ratings, samples):
    ret = 0.0
    numSamples = 0
    column = []
    for user, user_data in samples.items():
        # For each course in the user sample, generate the set of courses and 
        # the expected rank for the cousre
        for course_id, rank_range in user_data['grade_ranges'].items():
            column.append((ratings[course_id,user], rank_range))
        # Sort the list by grade
        column = sorted(column, key=lambda x: x[0], reverse=True)
        print(column)
        for item in column:
            ret += rank_difference(item[0],item[1]) ** 2
            numSamples += 1
    # for user, user_data in samples.items():
    #     # set_trace()
    #     column = ratings[:,user].tolist()
    #     # print(column)
    #     column = [(i,rating) for i, rating in enumerate(column)]
    #     # column = list(filter(lambda x: x[1] != 0.0, column))
    #     column = sorted(column, key=lambda x: x[1], reverse=True)
    #     # print(column)

    #     for rank,grade in enumerate(column):
    #         ret += rank_difference(rank,user_data['grade_ranges'][grade[0]]) ** 2
    #         numSamples += 1
    ret /= numSamples
    ret = sqrt(ret)
    return ret

# Given: List of ratings representing an MxN sparse matrix
# Returns: MxN matrix of rating approximations
def sgd(ratings, user_indexes, course_indexes, max_k=25, alpha=0.01, regularizer=0.02, default=0.1, max_iterations=100, debug=False):
    _ratings = deepcopy(ratings)

    U = np.full((max_k,len(user_indexes)), default)
    V = np.full((max_k,len(course_indexes)), default)

    for feature in np.arange(max_k):
        prev_norm_U = float('Inf')
        prev_norm_V = float('Inf')
        for iteration in np.arange(max_iterations):
            if debug:
                print("k: %d, iteration: %d", (feature, iteration))
            for rating in _ratings:
                user_feature = U[feature,rating.user_id]
                course_feature = V[feature,rating.course_id]
                rating.error = rating.value - user_feature*course_feature
                U[feature,rating.user_id] += alpha * (rating.error*course_feature - regularizer*user_feature)
                V[feature,rating.course_id] += alpha * (rating.error*user_feature - regularizer*course_feature)

            norm_U = la.norm(U)
            norm_V = la.norm(V)
            if abs(norm_U - prev_norm_U) < EPSILON and abs(norm_V - prev_norm_V) < EPSILON:
                if debug:
                    print("Feature has converged.")
                break
            prev_norm_U = norm_U
            prev_norm_V = norm_V

        for rating in _ratings:
            rating.value = rating.error
    return np.transpose(np.transpose(U).dot(V))

def test_with_grade_samples():
    grade_samples = read_samples()
    ratings, user_indexes, course_indexes = get_ratings(grade_samples=grade_samples)
    for k in range(30):
        sgd_ratings = sgd(ratings, user_indexes, course_indexes,max_k=k, max_iterations=30)
        print(k,rmse(sgd_ratings,grade_samples))

def test_with_user_samples():
    user_samples = read_user_sample()
    ratings, user_indexes, course_indexes = get_ratings(user_samples=user_samples)
    sgd_ratings = sgd(ratings, user_indexes, course_indexes,max_k=15, max_iterations=50)
    print(ranking_error(sgd_ratings,user_samples))

def test_em_with_grade_samples():
    samples = read_samples()
    M = get_data(format="COO", grade_samples=samples)
    ratings = maximization_iteration_zeroes(M)
    print(rmse(ratings,samples))

def test_em_with_user_samples():
    samples = read_user_sample()
    M = get_data(format="COO", user_samples=samples)
    ratings = maximization_iteration_zeroes(M)
    print(ranking_error(ratings,samples))

if __name__ == '__main__':
    test_em_with_user_samples()