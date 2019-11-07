% HIFIE3  Hierarchical interpolative factorization for integral equations in 3D.
%
%    This is essentially the same as HIFIE2 but extended to 3D, compressing
%    first cells to faces, then faces to edges, and finally edges to points.
%
%    This implementation is suited for first-kind integral equations or matrices
%    that otherwise do not have high contrast between the diagonal and off-
%    diagonal elements. For second-kind integral equations, see HIFIE3X.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N).
%
%    See also HIFIE2, HIFIE2X, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV.

function F = hifie3(A,x,occ,rank_or_tol,pxyfun,opts)
  F = hifie3_base(A,x,occ,rank_or_tol,@hifie_id,pxyfun,opts);
end