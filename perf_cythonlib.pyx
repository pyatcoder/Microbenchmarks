import numpy as np
cimport cython
#cython: language_level=3
from libc.stdio cimport FILE, fopen, fprintf, fclose, sprintf
from numpy.random import default_rng
from numpy.linalg import matrix_power
rg = default_rng()


cdef long fib_core(long n):
    if n < 2:
        return n
    return fib_core(n-1) + fib_core(n-2)


def fib(long n):
    return fib_core(n)


## quicksort ##
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void qsort_kernel(double[:] a, long lo, long hi):
    cdef:
        long i, j
        double pivot
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


def qsort(double[:] a, long lo, long hi):
    qsort_kernel(a, lo, hi)

## randmatstat ##
def randmatstat(long t):
    cdef long n = 5
    cdef long i
    v = np.zeros(t)
    w = np.zeros(t)
    for i in range(t):
        a = rg.standard_normal((n, n))
        b = rg.standard_normal((n, n))
        c = rg.standard_normal((n, n))
        d = rg.standard_normal((n, n))
        P = np.hstack((a, b, c, d))
        Q = np.vstack((np.hstack((a, b)), np.hstack((c, d))))
        v[i] = np.trace(matrix_power(np.dot(P.T, P), 4))
        w[i] = np.trace(matrix_power(np.dot(Q.T, Q), 4))
    return np.std(v)/np.mean(v), np.std(w)/np.mean(w)

## randmatmul ##

def randmatmul(n):
    A = rg.random((n, n))
    B = rg.random((n, n))
    return A @ B

## mandelbrot ##

cdef double abs2(double complex z):
    return z.real*z.real +  z.imag*z.imag

cdef long mandel(double complex z):
    cdef:
        double complex c = z
        long maxiter = 80
    for n in range(maxiter):
        if abs2(z) > 4.:
            return n
        z = z*z + c
    return maxiter

cpdef void mandelperf(long[:,:] a):
    cdef long i, r
    for i in range(21):
        for r in range(26):
            a[i, r] = mandel((r - 20)/10 + (i - 10)/10*1j)


def pisum():
    cdef:
        double s = 0.0
        long j, k
    for j in range(500):
        s = 0.0
        for k in range(1, 10001):
            s += 1.0/(k*k)
    return s


cpdef void parse_int_core(long[:] datain, long[:] dataout):
    cdef:
        long i
        long num = datain.shape[0]
        char s[11]
        char c
        long n
    for i in range(num):
        sprintf(s, "%lx", datain[i])
        n = 0
        for j in range(11):
            c = s[j]
            if c == b'\0':
                break
            if c > b'9':
                n = 16 * n + c - 87
            else:
                n = 16 * n + c - 48
        dataout[i] = n

cpdef void printfd(long t):
    cdef:
        long i
        FILE *f
    f = fopen("/dev/null", "w")
    for i in range(t):
        fprintf(f, "%ld %ld\n", i, i+1)
    fclose(f)