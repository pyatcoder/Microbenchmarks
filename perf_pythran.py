import numpy as np
import time
from numpy.random import default_rng
rg = default_rng()
from perf_pythranlib import fib, qsort_kernel, randmatstat_core, mandelperf, pisum, parse_int, printfd


## randmatstat ##

def randmatstat(t):
    n = 5
    a = rg.standard_normal((t, n, n))
    b = rg.standard_normal((t, n, n))
    c = rg.standard_normal((t, n, n))
    d = rg.standard_normal((t, n, n))
    return randmatstat_core(t, a, b, c, d)

## randmatmul ##

def randmatmul(n):
    A = rg.random((n,n))
    B = rg.random((n,n))
    return np.dot(A,B)

def print_perf(name, time):
    print("pythran," + name + "," + str(time*1000))

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
    print_perf ("parse_integers", tmin)

    assert mandelperf().sum() == 14791
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        mandelperf()
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("userfunc_mandelbrot", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        lst = np.random.rand(5000)
        t = time.time()
        qsort_kernel(lst, 0, len(lst)-1)
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
