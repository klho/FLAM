FLAM (Fast Linear Algebra in MATLAB)
====================================

This MATLAB library implements various fast algorithms for certain classes of matrices with hierarchical low-rank block structure. Such matrices commonly arise in physical problems, including many classical integral and differential equations, and have appeared in the literature under an assortment of related names (e.g., H-, H2-, FMM, HODLR, HSS, HBS). Other application domains include multivariate statistics and uncertainty quantification (covariance matrices).

The primary purpose of this library is for personal prototyping, though it has been recognized that others may find it useful as well. Consequently, the algorithms do not contain all the latest features, but they can be considered reasonably complete; for example, most codes support full adaptivity.

It is also worth noting that we mainly use the interpolative decomposition (ID) for low-rank approximation. This is by no means the only choice, but we find it especially convenient due to its structure-preserving and numerical compression properties.

Currently implemented algorithms include:

- core routines:
  - tree construction
  - interpolative decomposition
  - fast spectral norm estimation
- dense matrix routines:
  - interpolative fast multipole method
  - recursive skeletonization
    - multiply
    - sparse extension (solve, least squares)
  - recursive skeletonization factorization
    - multiply
    - solve
    - Cholesky multiply/solve
    - determinant
    - diagonal extraction/inversion (matrix unfolding, sparse multiply/solve)
  - hierarchical interpolative factorization for integral equations
    - multiply
    - solve
    - Cholesky multiply/solve
    - determinant
    - diagonal extraction/inversion (matrix unfolding, sparse multiply/solve)
- sparse matrix routines:
  - multifrontal factorization
    - multiply
    - solve
    - Cholesky multiply/solve
    - determinant
    - diagonal extraction/inversion (matrix unfolding, sparse multiply/solve)
  - hierarchical interpolative factorization for differential equations
    - multiply
    - solve
    - Cholesky multiply/solve
    - determinant
    - diagonal extraction/inversion (matrix unfolding, sparse multiply/solve)

All algorithm directories contain extensive tests. Please refer to the individual source codes for reference information.

FLAM is freely available under the GNU GPLv3; for alternate licenses, please contact the author.