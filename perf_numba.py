import numpy as np
from numpy.linalg import matrix_power
from numpy.random import default_rng, PCG64
from numba import jit
import time
rg = default_rng()
x = PCG64()
f_pcg64 = x.ctypes.next_double
state_addr = x.ctypes.state_address


## fibonacci ##
@jit(nopython=True, cache=True)
def fib(n):
    if n < 2:
        return n
    return fib(n-1)+fib(n-2)

## quicksort ##
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def randmatstat_core(t, a, b, c, d):
    v = np.zeros(t)
    w = np.zeros(t)
    for i in range(t):
        P = np.hstack((a[i], b[i], c[i], d[i]))
        Q = np.vstack((np.hstack((a[i], b[i])), np.hstack((c[i], d[i]))))
        v[i] = np.trace(matrix_power(P.T @ P, 4))
        w[i] = np.trace(matrix_power(Q.T @ Q, 4))
    return np.std(v)/np.mean(v), np.std(w)/np.mean(w)

def randmatstat(t):
    n = 5
    a = rg.standard_normal((t, n, n))
    b = rg.standard_normal((t, n, n))
    c = rg.standard_normal((t, n, n))
    d = rg.standard_normal((t, n, n))
    return randmatstat_core(t, a, b, c, d)

## randmatmul ##
# https://docs.scipy.org/doc/numpy/reference/random/extending.html
@jit(nopython=True)
def pcg64_random(n, state):
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = f_pcg64(state)
    return out

@jit(nopython=True)
def randmatmul(n, state):
    a = pcg64_random(n, state)
    b = pcg64_random(n, state)
    return np.dot(a, b)

## mandelbrot ##
@jit(nopython=True, cache=True)
def abs2(z):
    return z.real*z.real +  z.imag*z.imag

@jit(nopython=True, cache=True)
def mandel(z):
    maxiter = 80
    c = z
    for n in range(maxiter):
        if abs2(z) > 4:
            return n
        z = z*z + c
    return maxiter

@jit(nopython=True, cache=True)
def mandelperf():
    a = np.empty((21, 26), dtype=np.int64)
    for i in range(21):
        for r in range(26):
            a[i, r] = mandel(complex((r - 20)/10, (i - 10)/10)) 
    return a

@jit(nopython=True, cache=True)
def pisum():
    sum = 0.0
    for j in range(500):
        sum = 0.0
        for k in range(1, 10001):
            sum += 1.0/(k*k)
    return sum


@jit(nopython=True, cache=True)
def hex(b):
    x = b // 16
    y = b % 16
    return x + 48 if x < 10 else x + 55, y + 48 if y < 10 else y + 55

@jit(nopython=True, cache=True)
def toint(x, y):
    a1 = x - 48 if x < 58 else x - 55
    a2 = y - 48 if y < 58 else y - 55
    return a1 * 16 + a2

@jit(nopython=True, cache=True)
def int2hex(a):
    t = a.shape[0]
    u = np.empty(8 * t, dtype=np.int32)
    a8 = np.frombuffer(a, dtype=np.uint8)
    for i in range(t):
        for j in range(4):
            u[8 * i + 6 - 2 * j], u[8 * i + 7 - 2 * j] = hex(a8[4 * i + j])
    return u

@jit(nopython=True, cache=True)
def hex2int(v):
    t = v.shape[0] // 8
    b8 = np.empty(4 * t, dtype=np.uint8)
    for i in range(t):
        for j in range(4):
            b8[4 * i + j] = toint(v[8 * i + 6 - 2 * j], v[8 * i + 7 - 2 * j])
    return np.frombuffer(b8, dtype=np.uint32)

def parse_int(t):
    a = np.random.randint(0, 2 ** 32 - 1, t, dtype=np.uint32)
    u = int2hex(a)
    s = np.frombuffer(u, dtype='<U8')
    v = np.frombuffer(s, dtype=np.int32)
    b = hex2int(v)
    assert (a == b).all()
    return b[t - 1]


@jit(nopython=True, cache=True)
def int2ascii(n, asc):
    d = 0
    while n > 0:
        asc[d] = n % 10 + 48
        n = n // 10
        d += 1
    return d


@jit(nopython=True, cache=True)
def printfd_core(buf, start, t, buf_size):
    num = 0
    asc = np.empty(20, dtype=np.int8)
    i = start
    while i < t:
        d = int2ascii(i, asc)
        for j in range(d - 1, -1, -1):
            buf[num] = asc[j]
            num += 1
        buf[num] = 32
        num += 1
        d = int2ascii(i + 1, asc)
        for j in range(d - 1, -1, -1):
            buf[num] = asc[j]
            num += 1
        buf[num] = 10
        num += 1
        i += 1
        if num > buf_size:
            break
    return num, i

def printfd(t):
    buf_size = 10000
    buf = np.empty(buf_size + 50, dtype='u1')
    start = 1
    with open("/dev/null", "wb") as f:
    # with open("test.txt", "wb") as f:  #テスト用
        while start < t:
            num, start = printfd_core(buf, start, t, buf_size)
            f.write(buf[:num].tobytes())


def print_perf(name, time):
    print("numba," + name + "," + str(time*1000))

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
        lst = rg.random(5000)
        t = time.time()
        qsort_kernel(lst, 0, len(lst)-1)
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("recursion_quicksort", tmin)

    tmin = float('inf')
    for i in range(mintrials):
        lst = rg.random(5000)
        t = time.time()
        lst.sort()
        t = time.time()-t
        if t < tmin: tmin = t
    print_perf ("quicksort", tmin)

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
        C = randmatmul(1000, state_addr)
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
