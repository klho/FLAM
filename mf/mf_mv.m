% MF_MV  Multiply using multifrontal factorization.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N) in 1D and
%    O(N^(2*(1 - 1/D))) in D dimensions.
%
%    Y = MF_MV(F,X) produces the matrix Y by applying the factored matrix F to
%    the matrix X.
%
%    Y = MF_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default), Y = F.'*X
%    if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also MF2, MF3, MF_CHOLMV, MF_CHOLSV, MF_SV, MFX.

function X = mf_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', X = conj(mf_mv(F,conj(X),'c')); return; end

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if trans == 'n', X = mf_mv_nn(F,X);
    else,            X = mf_mv_nc(F,X);
    end
  elseif F.symm == 'h', X = mf_mv_h(F,X);
  elseif F.symm == 'p', X = mf_mv_p(F,X);
  end
end