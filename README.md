# Microbenchmarks

This is a collection of micro-benchmarks used to compare Julia's performance against
that of other languages.
It was formerly part of the Julia source tree.
The results of these benchmarks are used to generate the performance graph on the
[Julia homepage](https://julialang.org) and the table on the
[benchmarks page](https://julialang.org/benchmarks).

## Running benchmarks

This repository assumes that Julia has been built from source and that there exists
an environment variable `JULIAHOME` that points to the root of the Julia source tree.
This can also be set when invoking `make`, e.g. `make JULIAHOME=path/to/julia`.

To build binaries and run the benchmarks, simply run `make`.
Note that this refers to GNU Make, so BSD users will need to run `gmake`.

## Included languages:

* C
* Fortran
* Go
* Java
* JavaScript
* Julia
* LuaJIT
* Mathematica
* Matlab
* Python
* R
* Rust
* Scala
* Stata

## Python 関係で以下のコードを追加

- python (perf_python.py)  
perf.py を random で numpy 1.17.0 で導入された PCG64 を使用するように修正。numpy は、>= 1.17.0 をインストールしてください。 
- pypy (perf_pypy.py)  
perf_python.py とコードは同じ
- numba (perf_numba.py)
- cython (perf_cython.py, perf_cythonlib.pyx)
- pythran (perf_pythran.py, perf_pythranlib.py)

## Cython 及び Pythran のコンパイルコマンド

make ファイルに Cython 及び Pythran のコンパイルコマンドを書いていないので、事前にコンパイルしておく必要があります。

コンパイルコマンドは、以下を使っています。

Cython 

```
cython -3 perf_cythonlib.pyx 
gcc-8 --shared -O3 -fPIC -I/usr/include/python3.7m -o perf_cythonlib.cpython-37m-x86_64-linux-gnu.so perf_cythonlib.c -lpython3.7m
```


Pythran

```
pythran -std=c++17 -O3 -DUSE_XSIMD -march=native perf_pythranlib.py
```

C++のコンパイラーは、g++-8 よりも clang++-8 を使ったほうが速かったので、~/.pythranrc を以下のように修正した

```~/.pythranrc
[compiler]
defines=
undefs=
include_dirs=
libs=
library_dirs=
cflags=-std=c++17 -fno-math-errno -w -fvisibility=hidden -fno-wrapv
ldflags=-fvisibility=hidden -Wl,-strip-all
blas=blas
CC=/usr/bin/clang-8
CXX=/usr/bin/clang++-8
ignoreflags=-Wstrict-prototypes
```