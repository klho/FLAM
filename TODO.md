Here are some ideas for potential new features:

- Enable optimizations for translation invariance, i.e., compressing/skeletonizing only once per level instead of once per block. This can yield significant computational and storage savings. More generally, translation invariance can be considered an extreme form of symmetry; how can arbitrary, perhaps more subtle, symmetries be specified and exploited?

- Add partial factorization support, e.g., terminating `rskelf` before it reaches the root. Recall that `rskelf`-type algorithms essentially work by reducing the size of the global dense matrix level by level. For some problems, however, the top levels may not be compressible (e.g., oscillations becoming high-frequency); for these, it would be useful to allow early termination and to provide access to the partially compressed matrix, which, e.g., the user can then solve by iteration, as well as to the multilevel interpolation operators to map the result back to the full space.

- Add more sophisticated parallel computing capabilities, for example through the `parfor` construct. Currently, the only parallel support is through BLAS multithreading.

- Port Victor Minden's `srskelf` (https://github.com/victorminden/strong-skel) and "modernize" as appropriate on merging into FLAM. This is essentially the factorization version of the FMM and achieves empirical linear complexity in all dimensions just like `hifie`, but with tighter rank control and a cleaner tree structure. `srskelf` can be considered the current state of the art for recursive skeletonization methods.

- Add support for domain periodicity. This requires generalizing how `hypoct` defines neighbors.

- Implement sublinear-time updating as outlined [V. Minden, A. Damle, K.L. Ho, L. Ying. A technique for updating hierarchical skeletonization-based factorizations of integral operators. Multiscale Model. Simul. 14 (1): 42–64, 2016](http://dx.doi.org/10.1137/15M1024500). Note that this would not really work in Octave due to copy-on-write unless we adopt a handle-based approach to storing the factorizations.