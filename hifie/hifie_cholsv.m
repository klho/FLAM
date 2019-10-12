% HIFIE_CHOLSV  Solve using generalized Cholesky factor from hierarchical
%               interpolative factorization for integral equations.
%
%    Typical complexity: about half that of HIFIE_SV.
%
%    Y = HIFIE_CHOLSV(F,X) produces the matrix Y by applying the inverse of the
%    generalized Cholesky factor C of the factored matrix F = C*C' to the matrix
%    X. Requires that F be computed with the Hermitian positive definite option.
%
%    Y = HIFIE_CHOLSV(F,X,TRANS) computes Y = C\X if TRANS = 'N' (default),
%    Y = C.'\X if TRANS = 'T', and Y = C'\X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_MV,
%    HIFIE_SV.

function Y = hifie_cholsv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % apply Cholesky inverse
  Y = rskelf_cholsv(F,X,trans);
end