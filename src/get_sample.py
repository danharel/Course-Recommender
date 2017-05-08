import numpy as np
import pickle
from itertools import chain
from collections import Counter

from get_data import get_data

N = 30
M = 4
GRADE_SAMPLE_FILENAME = '../data/samples.txt'
USER_SAMPLE_FILENAME = '../data/user_samples.txt'

def write_samples():
    X = get_data(format="COO")
    # samples = np.random.choice(X.tolist(), N, False)
    samples = ref_points = [[X.row[i],X.col[i],X.data[i]] for i in np.random.choice(range(len(X.data)),size=N,replace=False)]
    samples = np.array(samples)
    np.savetxt(GRADE_SAMPLE_FILENAME, samples, ['%d', '%d', '%.18e'])

def read_samples():
    return np.loadtxt(GRADE_SAMPLE_FILENAME, dtype=[('column',int),('row',int),('rating',float)])

def write_user_sample():
    X = get_data(format="COO").tocsr()

    # Determine the highest number of courses a student has taken
    max_len = 0
    for i in range(X.shape[1]):
        column = X[:,i]
        if column.nnz > max_len:
            max_len = column.nnz

    # Generate a set of users who've taken the most courses
    users = []
    for i in range(X.shape[1]):
        column = X[:,i]
        if column.nnz == max_len:
            users.append(i)

    # Pick N random users who've taken the most courses
    sample_list = {}
    users = [user for user in np.random.choice(users, N, False)]
    print(users)
    for user in users:
        sample_list[user] = {}
        # Convert the scipy sparse column vector into a python list of tuples
        # (course_id, grade)
        column = X[:,i]
        column = [(i,grade) for i,grade in enumerate(chain.from_iterable(column.todense().tolist()))]
        # Filter out the grades that are undefined (== 0.0)
        column = list(filter(lambda x: x[1] != 0.0, column))
        # Count up the number of occurances of each grade
        c = Counter(grade[1] for grade in column)
        # For each grade, generate the range for which a course should appear
        # E.g., if there are 6 courses which the student got a 4.00 in, then the
        # these courses should be in the range [0,6], etc.
        grades = [4.00, 3.67, 3.33, 3.00, 2.67, 2.33,2.00, 1.67, 1.33, 1.00, 0.67, 0.33]
        ranges = {}
        runningSum = 0
        for grade in grades:
            ranges[grade] = [runningSum, runningSum + c[grade]]
            runningSum += c[grade]
        course_grade_mapping = {}
        for course_id,grade in column:
            course_grade_mapping[course_id] = ranges[grade]
        sample_list[user]['grade_ranges'] = course_grade_mapping

        # Select M random indexes from the list of grades
        indexes_to_remove = np.random.choice(range(max_len),size=M,replace=False)
        # Map each index to its corresponding course_id
        sample_list[user]['samples_removed'] = list(map(lambda x: column[x][0],indexes_to_remove))        

    with open(USER_SAMPLE_FILENAME, 'wb+') as fp:
        pickle.dump(sample_list, fp)

def read_user_sample():
    with open(USER_SAMPLE_FILENAME, 'rb') as fp:
        return pickle.load(fp)

if __name__ == '__main__':
    print(read_user_sample())