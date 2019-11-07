% HIFIE2X  Hierarchical interpolative factorization for second-kind integral
%          equations in 2D.
%
%    This is the essentially the same as HIFIE2 but with certain modifications
%    that make it more suitable for second-kind integral equations or matrices
%    that otherwise have high contrast between the diagonal and off-diagonal
%    elements.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): quasilinear in N.
%
%    See also HIFIE2, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV.

function F = hifie2x(A,x,occ,rank_or_tol,pxyfun,opts)
  F = hifie2_base(A,x,occ,rank_or_tol,@hifie_idx,pxyfun,opts);
end