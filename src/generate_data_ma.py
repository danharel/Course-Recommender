import numpy as np
import numpy.ma as ma

# Number of courses
_M = 5
# Number of students
_N = 5
# density; approximate proportion of the matrix that's filled
# [0,1]
_d = .10

def random_indexes(M,N,num_indexes):
    num = 0
    while num < num_indexes:
        ret = ( np.random.randint(M), np.random.randint(N), np.random.sample()*4+1, )
        yield ret
        num += 1

def matrix(M=_M,N=_N,d=_d):
    # Generate random matrix with values from [0,1)
    A = np.random.random((M,N,))
    # Generate random matrix with values from [0,4)
    A *= 4
    # Generate random matrix with values from [1,5)
    A += 1
    # Mask out (1-d)*100% values
    A = ma.masked_where(np.random.random((M,N,)) > d, A)
    return A

def DOK(M=_M,N=_N,d=_d):
    A = {}
    for index in random_indexes(M,N,M*N*d):
        A[(index[0],index[1],)] = index[2]
    return A

def LIL(M=_M,N=_N,d=_d):
    A = np.empty((_M,), dtype=np.object_)
    A.fill([])
    for index in random_indexes(M,N,M*N*d):
        A[index[0]] = (index[1],index[2],)
    return A

def COO(M=_M,N=_N,d=_d):
    return list(random_indexes(M,N,M*N*d))

# Generates a new matrix.
# By default returns a numpy matrix
# Optional parameter to change the format to one of:
#   matrix (default), DOK, LIL, COO, CSR
def new_matrix(M=_M,N=_N,d=_d,format="matrix"):
    # return locals[format](M, N, d)
    return eval('%s(%d,%d,%f)' % (format,M,N,d))

if __name__ == '__main__':
    print(new_matrix())
    # print(new_matrix(format="DOK"))
    # print(new_matrix(format="LIL"))
    # print(new_matrix(format="COO"))