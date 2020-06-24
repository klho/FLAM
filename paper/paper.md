---
title: 'FLAM: Fast Linear Algebra in MATLAB - Algorithms for Hierarchical Matrices'
tags:
  - hierarchical matrices
  - fast multipole method
  - low rank
  - interpolative decomposition
  - recursive skeletonization
  - hierarchical interpolative factorization
  - integral equations
  - partial differential equations
authors:
  - name: Kenneth L. Ho
    orcid: 0000-0001-5450-4966
    affiliation: 1
affiliations:
 - name: Center for Computational Mathematics, Flatiron Institute
   index: 1
date: 23 June 2020
bibliography: paper.bib
---

# Summary

Many large matrices in science and engineering possess a special hierarchical low-rank structure that enables fast multiplication and inversion, among other fundamental operations. Such matrices commonly occur in physical problems, including the classical integral (IE) and differential equations (DE) of potential theory, as well as, for example, in kernel methods for high-dimensional statistics and machine learning. Owing to their ubiquity and practical importance, a wide variety of techniques for exploiting this structure have been developed, appearing in the literature under an assortment of related names such as the fast multipole method (FMM) [@greengard:1987:j-comput-phys], $\mathcal{H}$- and $\mathcal{H}^2$-matrices [@hackbusch:1999:computing], hierarchically semiseparable matrices [@xia:2010:numer-linear-algebra-appl], hierarchically block separable matrices [@gillman:2012:front-math-china], and hierarchical off-diagonal low-rank matrices [@ambikasaran:2013:j-sci-comput]. Many of these now also have accompanying open-source software packages: `FMMLIB` [@gimbutas:2015:commun-comput-phys], `H2Lib` (http://www.h2lib.org), `STRUMPACK` [@rouet:2016:acm-trans-math-softw; @ghysels:2016:siam-j-sci-comput], `HODLRlib` [@ambikasaran:2019:j-open-source-softw], `hm-toolbox` [@massei:2020:siam-j-sci-comput], etc.

`FLAM` is a MATLAB (and Octave-compatible) library in this same vein, implementing methods mostly in the "recursive skeletonization" style. Briefly, the core algorithms take as input a matrix implicitly defined by a function to generate arbitrary entries and whose rows/columns are associated with points in $\mathbb{R}^d$ (while we nominally target $d \leq 3$, there is no such explicit limitation). The induced geometry exposes the low-rank matrix blocks, which are then compressed or sparsified using the interpolative decomposition (ID) in a multilevel manner. Being written in MATLAB, the code is quite readable and easy to extend, especially for research purposes, though it may not be the most performant. Still, it is reasonably feature-complete; currently provided algorithms include, for dense IE-like matrices with dimensions $M \geq N$:

- `ifmm`: ID-based kernel-independent FMM for $O(M + N)$ multiplication [@martinsson:2007:siam-j-sci-comput];

- `rskel`: recursive skeletonization for inversion [@ho:2012:siam-j-sci-comput] and least squares [@ho:2014:siam-j-matrix-anal-appl] via extended sparse embedding; typical complexity of $O(M + N)$ for $d = 1$ and $O(M + N^{3(1 - 1/d)})$ for $d > 1$;

- `rskelf`: recursive skeletonization factorization (RSF) [@ho:2016a:comm-pure-appl-math] for generalized LU/Cholesky decomposition; same complexity as above but restricted to $M = N$; allows multiply/solve with matrix or Cholesky factors, determinant computation, selected inversion, etc.; and

- `hifie`: hierarchical interpolative factorization (HIF) for IEs [@ho:2016a:comm-pure-appl-math]; like RSF but with quasilinear complexity for all $d$.

Similarly, for sparse DE-like matrices, we have:

- `mf`: multifrontal factorization (MF); basically the sparse equivalent of RSF; and

- `hifde`: HIF for DEs [@ho:2016b:comm-pure-appl-math]; like MF but with quasilinear complexity for all $d$.

Most of these have previously been published though some are perhaps new (if but straightforward modifications or extensions of existing ones).

Each algorithm comes with extensive tests demonstrating its usage and performance. For instance, `rskelf/tests/ie_circle.m` solves a second-kind Laplace boundary IE on the unit circle. At default settings, the problem is characterized by a *fully dense* square matrix of size $N = 16384$, which takes about 60 s to factor using MATLAB's `lu` at a storage cost of 2 GB. In contrast, `rskelf` requires just 0.7 s to compute an LU-like approximation to $10^{-12}$ precision, a speedup of about 80$\times$; once computed, the factorization needs only 9 MB to store and can be used to execute solves in 0.04 s. Another exemplary test case is `hifde/tests/fd_square2.m`, which considers a Poisson DE on a very rough and high-contrast background field. Standard preconditioners such as `ichol` do not deal well with such severe ill-conditioning, but `hifde` remains highly effective, producing preconditioners that converge in just a handful of iterations. For more detailed performance analyses, we refer the reader to the cited literature above.

The provided functionality is somewhat similar to that offered by other software packages, but `FLAM` implements certain advanced methods like `hifie` and `hifde`, which to our knowledge are not publicly available elsewhere (though a close relative of `hifde` can be found in `STRUMPACK`). Furthermore, `FLAM` leverages a different core framework that we believe is considerably simpler and particularly well-suited to MATLAB's concise and readable style.

`FLAM` was originally created to support the computations in @ho:2016a:comm-pure-appl-math and @ho:2016b:comm-pure-appl-math, and has since been used in @liu:2016:dartmouth-coll, @wang:2016:new-jersey-inst-tech, @corona:2017:j-comput-phys, @fang:2017:j-fluid-mech, @jiang:2017:multiscale-model-simul, @minden:2017b:multiscale-model-simul, @fan:2019:j-comput-phys, @li:2019:eng-anal-bound-elem, @tian:2019:commun-math-sci, @wang:2019:j-sci-comput, and @askham:2020:adv-comput-math. It has also served as the starting point for various more sophisticated codes such as those described in @minden:2016:multiscale-model-simul, @li:2017:res-math-sci, @minden:2017:stanford-univ, @minden:2017a:multiscale-model-simul, and @feliu-faba:2020:commun-math-sci.

We hope that `FLAM` will be a valuable research tool for the broader scientific community to explore the use of fast matrix methods in their applications of interest as well as to prototype new algorithmic ideas and implementations.

`FLAM` is released under the GPLv3 license and can be accessed at [klho.github.io/FLAM](http://klho.github.io/FLAM).

# Acknowledgements

We thank Lexing Ying, Victor Minden, and Anil Damle for many fruitful discussions. This work was originally initiated while the author was at Stanford University under the partial support of the National Science Foundation (grant DMS-1203554).

# References