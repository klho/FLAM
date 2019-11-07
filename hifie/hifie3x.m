% HIFIE3X  Hierarchical interpolative factorization for second-kind integral
%          equations in 3D.
%
%    This is the essentially the same as HIFIE3 but with certain modifications
%    that make it more suitable for second-kind integral equations or matrices
%    that otherwise have high contrast between the diagonal and off-diagonal
%    elements.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): quasilinear in N.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV.

function F = hifie3x(A,x,occ,rank_or_tol,pxyfun,opts)
  F = hifie3_base(A,x,occ,rank_or_tol,@hifie_idx,pxyfun,opts);
end