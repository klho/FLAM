% HIFIE_CHOLMV   Multiply using generalized Cholesky factor from hierarchical
%                interpolative factorization for integral equations.
%
%    Y = HIFIE_CHOLMV(F,X) produces the matrix Y by applying the generalized
%    Cholesky factor C of the factored matrix F = C*C' to the matrix X. Requires
%    that F be computed with the Hermitian positive-definite option.
%
%    Y = HIFIE_CHOLMV(F,X,TRANS) computes Y = C*X if TRANS = 'N' (default),
%    Y = C.'*X if TRANS = 'T', and Y = C'*X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLSV, HIFIE_MV,
%    HIFIE_SV.

function Y = hifie_cholmv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(F.symm,'p'),'FLAM:hifie_cholmv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifie_cholmv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % apply Cholesky factor
  Y = rskelf_cholmv(F,X,trans);
end