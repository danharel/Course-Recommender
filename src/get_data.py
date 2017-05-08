import csv
import os 
import pickle

import numpy as np
import scipy.sparse as sparse

filename = "data.csv"
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", filename)

class Rating:
    def __init__(self, value, user, user_id, course, course_id):
        self.value = value
        self.true_value = value
        self.user = user
        self.course = course
        self.user_id = user_id
        self.course_id = course_id
        self.error = 0.0

    def __str__(self):
        return '%s/%s "%s"/%s %s' % (self.user, self.user_id, self.course, self.course_id, self.value)

    def __repr__(self):
        return '%s/%s "%s"/%s %s' % (self.user, self.user_id, self.course, self.course_id, self.value)

def get_score(grade):
    return {
        'A': 4.0,
        'A-': 3.67,
        'B+': 3.33,
        'B': 3.0,
        'B-': 2.67,
        'C+': 2.33,
        'C': 2.0,
        'C-': 1.67,
        'D+': 1.33,
        'D': 1.0,
        'D-': 0.67,
        'F+': 0.33,
        'F': 0.0
    }[grade]

def get_data(format="matrix", grade_samples=None, user_samples=None):
    users = set()
    courses = set()
    grades = set()

    if grade_samples is not None and user_samples is not None:
        raise ValueError("Only one of (grade_samples, user_samples) is allowed")

    sampleIndexes = None
    if grade_samples is not None:
        sampleIndexes = {(sample[0],sample[1]) for sample in grade_samples}
    if user_samples is not None:
        sampleIndexes = {(course_id, user_id) for user_id, user_data in user_samples.items() for course_id in user_data['samples_removed']}

    with open(data_path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            users.add(row[0])
            courses.add(row[1])
            grades.add(row[2])

    users = {k: v for v, k in enumerate(sorted(users))}
    courses = {k: v for v, k in enumerate(sorted(courses))}

    # This was used to get files containing the user-index and course-index
    # mappings for future reference
    # with open('users.json', 'w+') as f:
    #     json.dump(users, f)
    # with open('courses.json', 'w+') as f:
    #     json.dump(courses, f)

    rating_list = []
    M = np.zeros((len(courses),len(users)))
    with open(data_path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            course_index = courses[row[1]]
            user_index = users[row[0]]

            if sampleIndexes is not None and (course_index,user_index) in sampleIndexes:
                continue

            M[course_index,user_index] = get_score(row[2])

            rating_list.append(Rating(
                get_score(row[2]),
                row[0],
                user_index,
                row[1],
                course_index
                ))


    if format == "matrix":
        return M
    elif format == "COO":
        return sparse.coo_matrix(M)
    elif format == "rating_list":
        return rating_list, users, courses
    else:
        raise ValueError("format must be one of: matrix, COO, rating_list")

if __name__ == '__main__':
    get_data()