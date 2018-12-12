% HIFIER_MV  Multiply using rectangular hierarchical interpolative factorization
%            for integral operators.
%
%    Y = HIFIER_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = HIFIER_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also HIFIE2R, HIFIER_SV.

function Y = hifier_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifier_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % apply matrix
  Y = rskelfr_mv(F,X,trans);
end