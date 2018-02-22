% RSKELFR_SV   Solve using rectangular recursive skeletonization factorization.
%
%    Y = RSKELFR_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = RSKELFR_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also RSKELFR, RSKELFR_MV, RSKELFR_SVD, RSKELFR_SVL, RSKELFR_SVU.

function Y = rskelfr_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_sv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskelfr_sv(F,conj(X),'c'));
    return
  end

  % no transpose
  if strcmpi(trans,'n')
    X = rskelfr_svl(F,X,trans);
    Y = rskelfr_svd(F,X,trans);
    Y = rskelfr_svu(F,Y,trans);

  % conjugate transpose
  else
    X = rskelfr_svu(F,X,trans);
    Y = rskelfr_svd(F,X,trans);
    Y = rskelfr_svl(F,Y,trans);
  end
end