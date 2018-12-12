% RSKELFR_MV   Multiply using rectangular recursive skeletonization
%              factorization.
%
%    Y = RSKELFR_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = RSKELFR_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also RSKELFR, RSKELF_MVD, RSKELF_MVL, RSKELF_MVU, RSKELF_SV.

function Y = rskelfr_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskelfr_mv(F,conj(X),'c'));
    return
  end

  % no transpose
  if strcmpi(trans,'n')
    X = rskelfr_mvu(F,X,trans);
    Y = rskelfr_mvd(F,X,trans);
    Y = rskelfr_mvl(F,Y,trans);

  % conjugate transpose
  else
    X = rskelfr_mvl(F,X,trans);
    Y = rskelfr_mvd(F,X,trans);
    Y = rskelfr_mvu(F,Y,trans);
  end
end