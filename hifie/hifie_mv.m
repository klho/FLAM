% HIFIE_MV   Multiply using hierarchical interpolative factorization for
%            integral equations.
%
%    Y = HIFIE_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = HIFIE_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV,
%    HIFIE_SV.

function Y = hifie_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifie_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % apply matrix
  Y = rskelf_mv(F,X,trans);
end