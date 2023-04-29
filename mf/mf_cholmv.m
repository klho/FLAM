% MF_CHOLMV  Multiply using Cholesky factor from multifrontal factorization.
%
%    Typical complexity: about half that of MF_MV.
%
%    Y = MF_CHOLMV(F,X) produces the matrix Y by applying the Cholesky factor C
%    of the factored matrix F = C*C' to the matrix X. Requires that F be
%    computed with the Hermitian positive definite option.
%
%    Y = MF_CHOLMV(F,X,TRANS) computes Y = C*X if TRANS = 'N' (default),
%    Y = C.'*X if TRANS = 'T', and Y = C'*X if TRANS = 'C'.
%
%    See also MF2, MF3, MF_CHOLSV, MF_MV, MF_SV, MFX.

function Y = mf_cholmv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  assert(F.symm == 'p','FLAM:mf_cholmv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(mf_cholmv(F,conj(X),'c')); return; end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward/downward sweep
  if trans == 'n'
    for i = n:-1:1
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      Y(sk,:) = Y(sk,:) + f.E*Y(rd,:);
      Y(rd,:) = f.L*Y(rd,:);
    end
  else
    for i = 1:n
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      Y(rd,:) = f.L'*Y(rd,:);
      Y(rd,:) = Y(rd,:) + f.E'*Y(sk,:);
    end
  end
end