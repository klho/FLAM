% HIFIE2  Hierarchical interpolative factorization for integral equations in 2D.
%
%    The hierarchical interpolative factorization is an extension of the
%    recursive skeletonization factorization with additional dimensional
%    reduction to achieve improved complexities in 2D and above. Specifically,
%    in 2D, it compresses cells to edges as in RSKELF, then further compresses
%    edges to points. The output is an approximation of a hierarchically rank-
%    structured matrix A as a generalized LDU (or LDL/Cholesky) decomposition.
%
%    This implementation is suited for first-kind integral equations or matrices
%    that otherwise do not have high contrast between the diagonal and off-
%    diagonal elements. For second-kind integral equations, see HIFIE2X.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N).
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL) produces a factorization F of the matrix A
%    acting on the points X using tree occupancy parameter OCC and local
%    precision parameter RANK_OR_TOL. See HYPOCT and ID for details. Since no
%    proxy function is supplied, this simply performs a naive compression of all
%    off-diagonal blocks.
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL,PXYFUN) accelerates the compression using
%    the proxy function PXYFUN to capture the far field (both incoming and
%    outgoing). This is a function of the form
%
%      [KPXY,NBR] = PXYFUN(X,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor point indices (can be modified)
%      - X:    input points
%      - SLF:  block point indices
%      - L:    block node size
%      - CTR:  block node center
%
%    The relevant arguments will be passed in by the algorithm; the user is
%    responsible for handling them. See the examples for further details. If
%    PXYFUN is not provided or empty (default), then the code uses the naive
%    global compression scheme.
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options to
%    the algorithm. Valid options include:
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = INF). See HYPOCT.
%
%      - EXT: set the root node extent to [EXT(D,1) EXT(D,2)] along dimension D.
%             If EXT is empty (default), then the root extent is calculated from
%             the data. See HYPOCT.
%
%      - TMAX: ID interpolation matrix entry bound (default: TMAX = 2). See ID.
%
%      - RRQR_ITER: maximum number of RRQR refinement iterations in ID (default:
%                   RRQR_ITER = INF). See ID.
%
%      - SKIP: skip the additional dimension reductions on the first SKIP levels
%              (default: SKIP = 0). More generally, this can be a logical
%              function of the form SKIP(LVL,L) that specifies whether to skip a
%              particular reduction based on the current tree level LVL above
%              the bottom and node size L.
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N'). If
%              SYMM = 'N' or 'S', then local factors are computed using the LU
%              decomposition; if SYMM = 'H', the LDL decomposition; and if
%              SYMM = 'P', the Cholesky decomposition. Symmetry can reduce the
%              computation time by about a factor of two.
%
%      - VERB: display status info if VERB = 1 (default: VERB = 0). This prints
%              to screen a table tracking compression statistics through level.
%              Each level is indexed as 'L-D', where L indicates the tree level
%              and D the current dimensionality of that level. Special levels:
%              'T', tree sorting.
%
%    Primary references:
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. Comm. Pure Appl. Math. 69 (7):
%        1314-1353, 2016.
%
%    Other references:
%
%      E. Corona, P.-G. Martinsson, D. Zorin. An O(N) direct solver for
%        integral equations on the plane. Appl. Comput. Harmon. Anal. 38 (2):
%        284-317, 2015.
%
%    See also HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV, HYPOCT, ID, RSKELF.

function F = hifie2(A,x,occ,rank_or_tol,pxyfun,opts)
  F = hifie2_base(A,x,occ,rank_or_tol,@hifie_id,pxyfun,opts);
end