% HIFIE_MV  Multiply using hierarchical interpolative factorization for integral
%           equations.
%
%    Typical complexity: quasilinear in all dimensions.
%
%    Y = HIFIE_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = HIFIE_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV,
%    HIFIE_SV.

function X = hifie_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % apply matrix
  X = rskelf_mv(F,X,trans);
end