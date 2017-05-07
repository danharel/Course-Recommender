import numpy as np
import numpy.ma as ma
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

from generate_data import new_matrix
from get_data import get_data
from get_sample import read_samples

from pdb import set_trace
from math import sqrt

EPSILON = 0.0001

def rmse_coo(A,B,ref_points=None):
    ret = 0
    if ref_points:
        for point in ref_points:
            ret += (point[2] - B[point[0],point[1]]) ** 2
        ret /= len(ref_points)
    else:
        for i,j,v in zip(A.row,A.col,A.data):
            ret += (v - B[i,j]) ** 2
        ret /= len(A.data)
    return sqrt(ret)

def rmse(ratings,samples):
    ret = 0.0
    for sample in samples:
        ret += (sample[2] - ratings[sample[0],sample[1]]) ** 2
    ret /= len(samples)
    ret = sqrt(ret)
    return ret

def fill_masked_matrix(A,fill_value=0):
    for i,row in enumerate(ma.getmask(X)):
        for j,is_mask in enumerate(row):
            if is_mask:
                X[i,j] = fill_value

def DINEOF(A, start_k=1, max_k=35, fill_value=0, num_ref_points=30):
    X = A.toarray()

    ref_points = {(A.row[i],A.col[i],A.data[i]) for i in np.random.choice(range(len(A.data)),size=num_ref_points,replace=False)}

    rms_prev = float('Inf')
    rms_now = 0
    # k = 1
    k=start_k
    print_val = True
    while k <= max_k:
        while True:
            # We want to conserve all of the given data
            for i,j,v in zip(A.row,A.col,A.data):
                # ...except for the reference points
                # if not (i,j,v) in ref_points:
                X[i,j] = v
            rms_prev = rms_now

            U,S,V = sparse.linalg.svds(X, k=k)
            X = U.dot(np.diag(S)).dot(V)

            # U,S,V = np.linalg.svd(X)
            # X = U[:,:k].dot(np.diag(S[:k])).dot(V[:k,])
            # rms_now = rmse(A,X,ref_points=ref_points)
            rms_now = rmse_coo(A,X)
            print("k: %d; RMSE: %f" % (k, rms_now))
            if print_val:
                for point in ref_points:
                    print(point[2], X[point[0],point[1]])
                print_val = False
            if abs(rms_prev-rms_now) < EPSILON:
                break
        k+=1
    for point in ref_points:
        print(point[2], X[point[0],point[1]])
    return X

def maximization_iteration_zeroes(A, max_k=35):
    return DINEOF(A, start_k=max_k, max_k=max_k)

def scipy_svds(A, max_k=25, max_iter=50):
    B = A.copy()
    rms_prev = float('Inf')
    rms_now = 0
    iter_num = 0
    while abs(rms_prev-rms_now) > EPSILON and iter_num < max_iter:
        rms_prev = rms_now
        U,S,V = svds(B, k=max_k)
        tmp = U.dot(np.diag(S)).dot(V)
        for i,j in zip(A.row,A.col):
            B[i,j] = tmp[i,j]
        rms_now = rmse_coo(A,B)
        iter_num += 1
    # set_trace()
    return B, rms_now

def print_differences(A,B):
    for i,j,v in zip(A.row,A.col,A.data):
        print(v,B[i,j])

def residual(A,A_new):
    # set_trace()
    R = np.zeros(shape=A.shape)
    it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        if it[0] != 0:
            R[it.multi_index] = A[it.multi_index]-A_new[it.multi_index]
        it.iternext()
    return R

# http://www-personal.umich.edu/~tianxili/SVDwithMissing/MissingValueSVD.pdf
def stochastic_gradient_a(A, alpha=0.01,k=40, max_iter=100):
    U = np.random.uniform(low=1.0,high=5.0,size=(A.shape[0],k))
    V = np.random.uniform(low=1.0,high=5.0,size=(A.shape[1],k))
    # U = np.ones(shape=(A.shape[0],k))
    # V = np.ones(shape=(A.shape[1],k))
    A_new = U.dot(np.transpose(V))
    e = float("Inf")
    R = residual(A,A_new)

    for i in np.arange(max_iter):
        e_new = la.norm(R) ** 2
        if e_new > e:
            alpha /= 2

        U_tmp = U + 2*alpha * (R.dot(V))
        V_tmp = np.transpose(np.transpose(V) + (2*alpha * np.transpose(U)).dot(R))
        U = U_tmp
        V = V_tmp

        A_tmp = U.dot(np.transpose(V))
        R = residual(A,A_new)

        if la.norm(A_tmp - A_new) < 0.00001*la.norm(A_new):
            return A_tmp
        A_new = A_tmp
    return A_new

def regularize(A_var,B):
    return A_var/np.var(B)

# https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf
# A is a sparse matrix
def stochastic_gradient_b(A, alpha=0.01, k=40, max_iter=100):
    A_var = np.var(A.toarray())

    # Courses
    U = np.random.uniform(low=0.0,high=1.0,size=(A.shape[0],k))
    # Students
    V = np.random.uniform(low=1.0,high=5.0,size=(A.shape[1],k))
    # Previous error
    for i,j,v in zip(A.row,A.col,A.data):
        print(i,j,v)

        err_p = float("Inf")
        iter_num = 0
        while iter_num < max_iter:
            err = v - U[i].dot(V[j])
            U[i] += alpha*(err*V[j])
            V[j] += alpha*(err*U[i])
            # U[i] += alpha*(err*V[j] - regularize(A_var,U[i])*U[i])
            # V[j] += alpha*(err*U[i] - regularize(A_var,V[j])*V[j])

            if abs(err-err_p) < EPSILON:
                break
            err_p = err
            iter_num+=1

    return U.dot(np.transpose(V))

# Try This At Home
def stochastic_gradient_c(A, alpha=0.01, k=40, max_iter=100):
    pass

def test_DINEOF():
    # print(A)
    A = new_matrix(format="sp_rand")
    X = DINEOF(A)
    # print(X)
    # print_differences(A,X)
    print(X)
    # print(scipy_svds(A))

def test_EM():
    samples = read_samples()
    M = get_data(format="COO", samples=samples)
    ratings = maximization_iteration_zeroes(M)
    # print(maximization_iteration_zeroes(M))
    print(rmse(ratings,samples))

def test_sgd_a():
    A = new_matrix(format="matrix")
    X = stochastic_gradient(A)
    print(X)

def test_sgd_b():
    A = new_matrix(format="sp_rand")
    X = stochastic_gradient_b(A)

    for i,j,v in zip(A.row,A.col,A.data):
        print(v,X[i,j])

if __name__ == '__main__':
    test_EM()