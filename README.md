FLAM (Fast Linear Algebra in MATLAB): Algorithms for Hierarchical Matrices
==========================================================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1253581.svg)](https://doi.org/10.5281/zenodo.1253581) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01906/status.svg)](https://doi.org/10.21105/joss.01906)

This MATLAB (and Octave-compatible) library implements various fast algorithms for certain classes of matrices with hierarchical low-rank block structure. Such matrices commonly arise in physical problems, including many classical integral (IE) and differential equations (DE), as well as in statistical contexts such as uncertainty quantification and machine learning. They have appeared in the literature under an assortment of related names and frameworks (e.g., H-, H2-, FMM, HODLR, HSS, HBS), where their special properties have been used to develop highly accelerated algorithms for many fundamental linear algebraic operations.

FLAM is an implementation of some of these ideas, following mostly the "recursive skeletonization" approach as outlined in:

- P.G. Martinsson, V. Rokhlin. A fast direct solver for boundary integral equations in two dimensions. J. Comput. Phys. 205 (1): 1-23, 2005.
- K.L. Ho, L. Greengard. A fast direct solver for structured linear systems by recursive skeletonization. SIAM J. Sci. Comput. 34 (5): A2507-A2532, 2012.
- K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic operators: integral equations. Comm. Pure Appl. Math. 69 (7): 1314-1353, 2016.
- K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic operators: differential equations. Comm. Pure Appl. Math. 69 (8): 1415-1451, 2016.
- V. Minden, K.L. Ho, A. Damle, L. Ying. A recursive skeletonization factorization based on strong admissibility. Multiscale Model. Simul. 15 (2): 768-796, 2017.

It was originally intended as a research testbed for personal prototyping, though it is now emerging as a more general community resource. As such, the implementations do not necessarily contain all the latest features, but they can be considered reasonably complete; for example, most codes support full geometric adaptivity. Currently available algorithms include:

- Dense IE-like matrices:
  - `ifmm`: interpolative fast multipole method
  - `rskel`: recursive skeletonization
  - `rskelf`: recursive skeletonization factorization
  - `hifie`: hierarchical interpolative factorization for IEs
- Sparse DE-like matrices:
  - `mf`: multifrontal factorization
  - `hifde`: hierarchical interpolative factorization for DEs

See [Algorithms](#algorithms) for details.

FLAM has been written to emphasize readability and ease of use. This follows the choice of MATLAB as the computing platform, and we have preferred standard idiomatic MATLAB over more complicated constructions where possible. A key goal is for the codes to be readily understood at an introductory graduate level so that they may be broadly deployed, modified, and tested in an effort to make such fast matrix methods more accessible. Of course, this design may not yield the best performance; a more serious undertaking would re-implement the algorithms in a lower-level language&mdash;but this is beyond the scope of FLAM.

FLAM is freely available under the GNU GPLv3 and was last tested on MATLAB R2019a and Octave 4.2.2.

**Note**: Although FLAM is technically compatible with Octave, its practical performance may lag behind that in MATLAB, even in terms of asymptotic scaling. This is because certain functions make use of MATLAB's facilities for in-place computation, which Octave does not yet support. Despite this admittedly very significant drawback, we have chosen to write the code in this way for reasons of clarity and modularity.

If you use FLAM, please cite:

- K.L. Ho. FLAM: Fast linear algebra in MATLAB &ndash; Algorithms for hierarchical matrices. J. Open Source Softw. 5 (51): 1906, 2020. [doi:10.21105/joss.01906](http://dx.doi.org/10.21105/joss.01906).

## Contents

- [Getting started](#getting-started)
- [Algorithms](#algorithms)
  - [Dense matrices](#dense-matrices)
    - [Interpolative fast multipole method](#interpolative-fast-multipole-method)
    - [Recursive skeletonization](#recursive-skeletonization)
    - [Recursive skeletonization factorization](#recursive-skeletonization-factorization)
    - [Hierarchical interpolative factorization for integral equations](#hierarchical-interpolative-factorization-for-integral-equations)
  - [Sparse matrices](#sparse-matrices)
    - [Multifrontal factorization](#multifrontal-factorization)
    - [Hierarchical interpolative factorization for differential equations](#hierarchical-interpolative-factorization-for-differential-equations)
- [Supporting functions](#supporting-functions)
- [Bug reports](#bug-reports)
- [Contributing](#contributing)

## Getting started

Download FLAM by typing, e.g.:

```sh
git clone https://github.com/klho/FLAM
```

This will create a local directory called `FLAM` containing the contents of the entire repository. Then enter the `FLAM` directory and launch MATLAB. This will automatically run FLAM's startup script and set up all required paths. Alternatively, if you launched MATLAB from a different directory or are using Octave instead, do this manually by running `startup.m` inside `FLAM`.

All FLAM functions should now be accessible. Test this by typing, e.g.:

```matlab
help rskelf
```

which should display the documentation for `rskelf`:

```
RSKELF  Recursive skeletonization factorization.

    The recursive skeletonization factorization  % ...
```

Other functions are similarly documented.

All of the algorithms listed in the preamble are accompanied by extensive tests demonstrating their usage and performance. As an example, consider `ie_circle.m` inside `FLAM/rskelf/test`, which solves a second-kind boundary IE on the unit circle. Briefly, the IE is derived from an underlying Laplace DE:

    Δu(x) = 0     for  x ∊  Ω
     u(x) = f(x)  for  x ∊ ∂Ω

on the unit disk `Ω` by writing the solution as a double-layer potential `u(x) = D[σ](x)` over an unknown surface density `σ`, where `D` is a convolution against the normal derivative of the Green's function. This representation satisfies the DE in the interior by construction; matching the boundary condition then gives the IE `Aσ = f`, where, in operator notation, `A = -1/2*I + D` for `D` now understood in the principal value sense. The matrix `A` is *fully dense* and thus can be very challenging to solve directly using conventional techniques. The purpose of this example is to demonstrate that in fact we can do so very efficiently, in linear time, by exploiting the inherent structure in `A`.

As with all tests, `ie_circle` is actually written as a function, with parameters to set the number of discretization points, compression accuracy, symmetry properties, etc.; sensible defaults are provided for any omitted arguments. Please see the test file for more information.

Running `ie_circle` then gives, e.g.:

```
>> cd rskelf/test
>> ie_circle
---------------------------------------------------------------------
lvl |   nblk |      start/end npts |  start/end npts/blk |   time (s)
---------------------------------------------------------------------
  t |                                                        1.36e-01
  8 |    492 |    16384 |     8323 |    33.30 |    16.92 |   2.47e-01
  7 |    252 |     8323 |     4042 |    33.03 |    16.04 |   1.92e-01
  6 |    124 |     4042 |     2290 |    32.60 |    18.47 |   8.01e-02
  5 |     60 |     2290 |     1021 |    38.17 |    17.02 |   4.18e-02
  4 |     28 |     1021 |      521 |    36.46 |    18.61 |   1.84e-02
  3 |     12 |      521 |      266 |    43.42 |    22.17 |   8.90e-03
  2 |      4 |      266 |       20 |    66.50 |     5.00 |   3.92e-03
  1 |      1 |       20 |        0 |    20.00 |     0.00 |   8.27e-04
---------------------------------------------------------------------
rskelf time/mem: 7.3356e-01 (s) /   8.86 (MB)
rskelf_mv err/time: 2.3835e-12 / 3.6486e-02 (s)
rskelf_sv err/time: 2.3127e-12 / 3.7973e-02 (s)
pde solve err: 1.3504e-12
```

The actual numbers may differ from run to run but should be roughly the same as those listed here. Let's now walk through this test in order and explain each output:

- Associated with the IE is a dense square matrix, which in this case has order 16384. First, we use `rskelf` to compress and factorize this matrix in a multilevel manner; the table at the top shows various compression statistics through level. In particular, we see, from left to right: the current tree level (relative to the root at `lvl = 1`), the number of blocks on that level, the total number of points (i.e., row/column indices) before/after compression, the average number of points per block before/after compression, and the time taken. The output of `rskelf` is a generalized LU decomposition, which in total requires about 0.7 s to compute (vs. 58 s for dense LU) and 8.86 MB to store (vs. 2 GB dense).

- We then validate the factorization by checking the forward and inverse apply errors. The default target precision is 1e-12, which both successfully achieve; see the test file for details. Applying the factorized matrix and its inverse each take 0.03 - 0.04 s.

- Finally, we use the factorization to solve an actual instance of the IE and compare the result against a known analytical solution. This is done by constructing the right-hand side as the restriction of a known Laplace field to the boundary, solving for the layer density, then using that to evaluate the field in the volume through the double-layer potential. The match between the exact and numerically computed fields is quite good.

Pay close attention to the number of "skeleton" points per block at the end of each level, which remains roughly constant as we move up the tree. At the same time, the number of blocks decreases geometrically; this forms the basis of `rskelf`'s linear computational complexity when applied to quasi-1D problems (e.g., boundary IEs in 2D). Indeed, running at some larger sizes gives:

```
>> ie_circle(2^17)  % 2^17 = 131,072
% ...
rskelf time/mem: 5.6300e+00 (s) /  80.90 (MB)
% ...

>> ie_circle(2^20)  % 2^20 = 1,048,576
% ...
rskelf time/mem: 4.8768e+01 (s) / 750.45 (MB)
% ...
```

Perfect linear scaling is observed. However, this property does not extend to higher dimensions, for which more advanced algorithms like `hifie` may be better suited; see [Algorithms](#algorithms) for further information.

Many other tests beyond `ie_circle` are also available, including IEs with different kernels and on various other geometries (prefixed with `ie_*`), sparse finite difference discretizations of DEs (`fd_*`), and kernel covariance matrices (`cov_*`). The same general style is followed for tests belonging to other algorithms as well. From this, you should now be able to navigate FLAM's directory structure and to understand the usage and performance characteristics of the different algorithms by running the tests and&mdash;when in doubt&mdash;reading the source.

## Algorithms

FLAM contains a variety of methods for working with structured matrices. All are fundamentally based on using geometry to reveal rank structure and so require this information to be passed in by attaching a spatial coordinate to each matrix index. Other common features include optimizations for matrix symmetry (which the user must specify since we cannot afford to check) and a tunable accuracy parameter.

The algorithms can largely be divided into two groups: those designed for dense matrices and those for sparse ones. The former is nominally more complicated but somewhat more generic and hence easier to explain&mdash;so we'll start there. Throughout, let `M >= N` denote the matrix dimensions and let `d` be the intrinsic dimension of the corresponding spatial geometry. While the algorithms are mostly designed for `d <= 3`, there is no such explicit limitation in the code.

### Dense matrices

There are two main objectives when dealing with dense matrices. The first is *compression*, by which we mean simply beating the naive quadratic storage complexity by reducing to a more compact data-sparse form. Such compressed representations are typically written as additive low-rank perturbations. They enable fast multiplication but generally do not extend to inversion or other more complex operations (of course, they can still drive fast iterative solvers).

This is in contrast to *factorization*, which employs instead a multiplicative structure as in a generalized LU decomposition. Fast direct inversion is naturally supported, as are other capabilities facilitated by the standard LU, e.g., determinant computation and Cholesky square roots, where applicable. Factorization is often more complicated than compression because of the extra internal structure required and consequently may impose more restrictions on the input matrix.

The matrix itself is passed in as a function handle in order to avoid the cost of generating and storing it full. Similarly, we will often need to compress, in principle, an entire off-diagonal block row/column, which is global in extent; the user can provide a "proxy" function&mdash;utilizing, for instance, analytic properties of the underlying matrix kernel&mdash;to localize and accelerate this step. The interpolative decomposition (ID) is used for all low-rank approximation since it has a special structure-preserving property that is critical for high efficiency.

#### Interpolative fast multipole method

`ifmm` is a purely numerical kernel-independent fast multipole method (FMM) for linear-time matrix compression and multiplication. Compared to conventional analytic FMMs, it is substantially more general but can be more expensive, especially during the initial compression, since it has to essentially discover the translation operators bridging across levels. The near-field can optionally be compressed and various storage modes are available to trade memory for speed.

Available functions:

- `ifmm`: main routine for matrix compression
- `ifmm_mv`: apply compressed matrix to a vector

#### Recursive skeletonization

While `ifmm` can be effective for solving many linear systems by iteration, certain challenging environments (e.g., ill-conditioning, multiple right-hand sides, updating) more naturally demand direct solution techniques. The complicated near- and far-field structure of `ifmm`, however, does not lend itself easily to this task. `rskel` was developed in partial response to this need, sacrificing efficiency for a simpler structure that can be leveraged into a fast direct solver. Indeed, the compression algorithm now has linear complexity only in 1D, with computational and storage costs of `O(M + N^(3(1 - 1/d)))` and `O(M + N^(2(1 - 1/d)))`, respectively, for `d > 1`. Once the matrix `A` has been compressed, its factors can be rearranged in a special way and embedded into an extended sparse matrix `A_xsp` such that `A_xsp` can be efficiently factored and the solution of `A*x = b` can be recovered from that of the corresponding extended system `A_xsp*x_xsp = b_xsp`. Variants of this idea also allow the solution of least squares problems with rectangular `A` (see `rskel/test/{ols,uls}_*` for full-rank over- and under-determined examples, respectively).

Available functions:

- `rskel`: main routine for matrix compression
- `rskel_mv`: apply compressed matrix to a vector
- `rskel_xsp`: embed compressed matrix into extended sparse form

#### Recursive skeletonization factorization

`rskelf` is a reformulation of `rskel` for square matrices in terms of a multiplicative factorization. This effectively combines the separate compression and (sparse) factorization steps into one and bypasses the need to dispatch to external sparse direct solvers. It has the same complexity as `rskel` but typically with better constants. The form of the factorization itself also supports more immediate functionality, e.g., the generalized LU decomposition naturally becomes a generalized Cholesky decomposition when the matrix is positive definite.

Available functions:

- `rskelf`: main routine for matrix factorization
- `rskelf_mv`: apply factored matrix to a vector
- `rskelf_sv`: apply factored matrix inverse to a vector
- `rskelf_cholmv`: apply factored matrix Cholesky square root to a vector (positive definite only)
- `rskelf_cholsv`: apply factored matrix Cholesky square root inverse to a vector (positive definite only)
- `rskelf_logdet`: compute log-determinant of factored matrix
- `rskelf_diag`: extract diagonal of factored matrix or its inverse by "matrix unfolding"
- `rskelf_partial_info`: retrieve compressed skeleton information from partial factorization
- `rskelf_partial_mv`: apply partially factored matrix to a vector
- `rskelf_partial_sv`: apply partially factored matrix inverse to a vector
- `rskelf_spdiag`: extract diagonal of factored matrix or its inverse by sparse multiply/solves

#### Hierarchical interpolative factorization for integral equations

`hifie` is an extension of `rskelf` with improved performance in 2D (`hifie2`) and 3D (`hifie3`). It does this through a recursive dimensional reduction strategy that effectively reduces all problems to 1D, thereby achieving estimated linear complexity. Some special care is required, however, for second-kind IEs or matrices that otherwise have large contrasts between the diagonal and off-diagonal blocks in order to retain accuracy. Such modifications are made in `hifie{2,3}x`, at the price of slightly increased quasilinear costs. All of these construct a generic `hifie` factorization that can then be used as input to the other routines.

Available functions:

- `hifie2`: matrix factorization in 2D
- `hifie2x`: matrix factorization in 2D (second-kind IEs)
- `hifie3`: matrix factorization in 3D
- `hifie3x`: matrix factorization in 3D (second-kind IEs)
- `hifie_mv`: apply factored matrix to a vector
- `hifie_sv`: apply factored matrix inverse to a vector
- `hifie_cholmv`: apply factored matrix Cholesky square root to a vector (positive definite only)
- `hifie_cholsv`: apply factored matrix Cholesky square root inverse to a vector (positive definite only)
- `hifie_logdet`: compute log-determinant of factored matrix
- `hifie_diag`: extract diagonal of factored matrix or its inverse by "matrix unfolding"
- `hifie_spdiag`: extract diagonal of factored matrix or its inverse by sparse multiply/solves

### Sparse matrices

Our sparse matrix algorithms are essentially just optimized versions of the dense algorithms above for the special case where the input has only local interactions. In fact, the dense codes can be used without modification on such sparse matrices, though they simply will not be competitive in terms of performance. Sparsity optimizations are therefore critical, which can be quite involved. Still, a few important simplifications are afforded over the dense setting: first, sparse matrices are already compressed by definition, so we may focus only on factorization; and second, no proxy functions are needed since all data are local in scope.

#### Multifrontal factorization

`mf` is the standard sparse LU/Cholesky decomposition based on the classical nested dissection ordering. A general implementation is provided by `mfx`; this is basically the sparse equivalent of `rskelf`, in which skeletonization (meaning compression plus elimination) is replaced by elimination only, and can handle arbitrary meshes and interactions. Specialized optimizations for the most common use cases of nearest-neighbor stencils on regular meshes in 2D and 3D are available in `mf{2,3}`. Note that these all do not entail any low-rank approximation and so are formally exact, hence there is no accuracy input.

Available functions:

- `mf2`: matrix factorization on regular meshes in 2D
- `mf3`: matrix factorization on regular meshes in 3D
- `mfx`: matrix factorization on general meshes
- `mf_mv`: apply factored matrix to a vector
- `mf_sv`: apply factored matrix inverse to a vector
- `mf_cholmv`: apply factored matrix Cholesky square root to a vector (positive definite only)
- `mf_cholsv`: apply factored matrix Cholesky square root inverse to a vector (positive definite only)
- `mf_logdet`: compute log-determinant of factored matrix
- `mf_diag`: extract diagonal of factored matrix or its inverse by "matrix unfolding"
- `mf_spdiag`: extract diagonal of factored matrix or its inverse by sparse multiply/solves

#### Hierarchical interpolative factorization for differential equations

Likewise, `hifde` is the sparse counterpart to `hifie`, where the initial cell skeletonization on each level is replaced by simple elimination. As with `mf`, `hifde` comes in several flavors: `hifde2` for regular meshes in 2D and a more general version in `hifde2x`, both with linear complexity; `hifde3` for regular meshes in 3D, encompassing cell elimination and face skeletonization at `O(N*log(N))` cost; and `hifde3x` for arbitrary meshes and interactions in 3D, additionally instituting edge skeletonization in order to restore `O(N)` scaling.

Available functions:

- `hifde2`: matrix factorization on regular meshes in 2D
- `hifde2x`: matrix factorization on general meshes in 2D
- `hifde3`: matrix factorization on regular meshes in 3D (no edge skeletonization)
- `hifde3x`: matrix factorization on general meshes in 3D
- `hifde_mv`: apply factored matrix to a vector
- `hifde_sv`: apply factored matrix inverse to a vector
- `hifde_cholmv`: apply factored matrix Cholesky square root to a vector (positive definite only)
- `hifde_cholsv`: apply factored matrix Cholesky square root inverse to a vector (positive definite only)
- `hifde_logdet`: compute log-determinant of factored matrix
- `hifde_diag`: extract diagonal of factored matrix or its inverse by "matrix unfolding"
- `hifde_spdiag`: extract diagonal of factored matrix or its inverse by sparse multiply/solves

## Supporting functions

In addition to the main algorithms listed above, several core supporting functions may also be of independent interest:

- `hypoct`: adaptive point hyperoctree construction
- `id`: interpolative decomposition with rank-revealing QR refinement
- `snorm`: fast spectral norm estimation by randomized power method

These are all located under the `core` subdirectory. Please see the source files for details.

## Bug reports

If you find a bug, please follow the standard GitHub procedure and create a new issue. Be sure to provide sufficient information for the bug to be reproduced.

## Contributing

Contributions are welcome and can be made via pull requests. For some ideas on potential features to work on, see `TODO.md`.