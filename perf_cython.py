from numpy import zeros, empty, concatenate, trace, std, mean, dot, int64
from numpy.linalg import matrix_power
import time
from numpy.random import default_rng
from perf_cythonlib import fib, qsort, mandelperf, pisum, parse_int_core, printfd
rg = default_rng()


## randmatstat ##

def randmatstat(t):
    n = 5
    v = zeros(t)
    w = zeros(t)
    a = rg.standard_normal((t, n, n))
    b = rg.standard_normal((t, n, n))
    c = rg.standard_normal((t, n, n))
    d = rg.standard_normal((t, n, n))
    for i in range(t):
        P = concatenate((a[i], b[i], c[i], d[i]), axis=1)
        Q = concatenate((concatenate((a[i], b[i]), axis=1), concatenate((c[i], d[i]), axis=1)), axis=0)
        v[i] = trace(matrix_power(dot(P.T,P), 4))
        w[i] = trace(matrix_power(dot(Q.T,Q), 4))
    return (std(v)/mean(v), std(w)/mean(w))

## randmatmul ##

def randmatmul(n):
    A = rg.random((n,n))
    B = rg.random((n,n))
    return A @ B

def parse_int(t):
    a = rg.integers(0, 2 ** 32, size=t)
    data_out = empty(t, dtype=int64)
    parse_int_core(a, data_out)
    assert all(a == data_out)


def print_perf(name, time):
    print("cython," + name + "," + str(time*1000))

## run tests ##

if __name__=="__main__":

    mintrials = 5

    assert fib(20) == 6765
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        f = fib(20)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf("recursion_fibonacci", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        n = parse_int(1000)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf("parse_integers", tmin)

    a = empty((21, 26), dtype=int64)
    mandelperf(a)
    assert a.sum() == 14791
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        a = empty((21, 26), dtype=int64)
        mandelperf(a)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf("userfunc_mandelbrot", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        lst = rg.random(5000)
        t = time.time()
        qsort(lst, 0, len(lst)-1)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("recursion_quicksort", tmin)

    assert abs(pisum()-1.644834071848065) < 1e-6
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        pisum()
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("iteration_pi_sum", tmin)

    (s1, s2) = randmatstat(1000)
    assert s1 > 0.5 and s1 < 1.0
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        randmatstat(1000)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("matrix_statistics", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        C = randmatmul(1000)
        assert C[0,0] >= 0
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("matrix_multiply", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        printfd(100000)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("print_to_file", tmin)
