from numpy import zeros, concatenate, trace, std, mean, dot
from numpy.random import randn
from numpy.linalg import matrix_power
import time
import random
from numpy.random import default_rng
rg = default_rng()


## fibonacci ##

def fib(n):
    if n<2:
        return n
    return fib(n-1)+fib(n-2)

## quicksort ##

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

def mandelperf():
    r1 = [-2. + 0.1*i for i in range(26)]
    r2 = [-1. + 0.1*i for i in range(21)]
    return [mandel(complex(r, i)) for r in r1 for i in r2]

def pisum():
    sum = 0.0
    for j in range(1, 501):
        sum = 0.0
        for k in range(1, 10001):
            sum += 1.0/(k*k)
    return sum

def parse_int(t):
    for i in range(1,t):
        n = random.randint(0,2**32-1)
        s = hex(n)
        #s = string(n, base = 16)
        if s[-1]=='L':
            s = s[0:-1]
        m = int(s,16)
        assert m == n
    return n

def printfd(t):
    f = open("/dev/null", "w")
    for i in range(1,t):
        f.write("{:d} {:d}\n".format(i, i+1))
    f.close()


def print_perf(name, time):
    print("pypy," + name + "," + str(time*1000))

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

    assert sum(mandelperf()) == 14791
    tmin = float('inf')
    for i in range(mintrials):
        t = time.time()
        mandelperf()
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("userfunc_mandelbrot", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        lst = [random.random() for i in range(1,5000)]
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
