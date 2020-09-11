% HIFIE_SV  Solve using hierarchical interpolative factorization for integral
%           equations.
%
%    Typical complexity: same as HIFIE_MV.
%
%    Y = HIFIE_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = HIFIE_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV,
%    HIFIE_MV.

function X = hifie_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % apply matrix inverse
  X = rskelf_sv(F,X,trans);
end