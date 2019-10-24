import numpy as np
from numpy.random import randn

## fibonacci ##
#pythran export fib(int64)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1)+fib(n - 2)

## quicksort ##
#pythran export qsort_kernel(float64[:], int64, int64)
def qsort_kernel(a, lo, hi):
    i = lo
    j = hi
    while i < hi:
        pivot = a[(lo+hi) // 2]
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if lo < j:
            qsort_kernel(a, lo, j)
        lo = i
        j = hi
    return a


## randmatstat ##
#pythran export randmatstat_core(int64, float64[:,:,:] order(C), float64[:,:,:] order(C), float64[:,:,:] order(C), float64[:,:,:] order(C))
def randmatstat_core(t, a, b, c, d):
    v = np.zeros(t)
    w = np.zeros(t)
    for i in range(t):
        P = np.hstack((a[i], b[i], c[i], d[i]))
        Q = np.vstack((np.hstack((a[i], b[i])), np.hstack((c[i], d[i]))))
        v[i] = np.trace(np.linalg.matrix_power(np.dot(P.T, P), 4))
        w[i] = np.trace(np.linalg.matrix_power(np.dot(Q.T,Q), 4))
    return (np.std(v)/np.mean(v), np.std(w)/np.mean(w))


## randmatmul ##
#pythran export randmatmul(int64)
def randmatmul(n):
    A = np.random.rand(n,n)
    B = np.random.rand(n,n)
    return np.dot(A,B)


## mandelbrot ##
def abs2(z):
    return z.real*z.real +  z.imag*z.imag


def mandel(z):
    maxiter = 80
    c = z
    for n in range(maxiter):
        if abs2(z) > 4:
            return n
        z = z*z + c
    return maxiter

#pythran export mandelperf()
def mandelperf():
    a = np.empty((21, 26), dtype=np.int64)
    for i in range(21):
        for r in range(26):
            a[i, r] = mandel(complex((r - 20)/10, (i - 10)/10))
    return a

#pythran export pisum()
def pisum():
    sum = 0.0
    for j in range(1, 501):
        sum = 0.0
        for k in range(1, 10001):
            sum += 1.0/(k*k)
    return sum

#pythran export parse_int(int64)
def parse_int(t):
    a = np.random.randint(0, 2 ** 32 - 1, t)
    for c in a:
        s = hex(c)
        b = int(s, 16)
        assert c == b
    return a[t - 1]

#pythran export printfd(int64)
def printfd(t):
    f = open("/dev/null", "w")
    for i in range(1, t):
        f.write(str(i) + " " + str(i + 1) + "\n")
    f.close()
