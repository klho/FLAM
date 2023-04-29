% MF_CHOLSV  Solve using Cholesky factor from multifrontal factorization.
%
%    Typical complexity: about half that of MF_SV.
%
%    Y = MF_CHOLSV(F,X) produces the matrix Y by applying the inverse of the
%    Cholesky factor C of the factored matrix F = C*C' to the matrix X. Requires
%    that F be computed with the Hermitian positive definite option.
%
%    Y = MF_CHOLSV(F,X,TRANS) computes Y = C\X if TRANS = 'N' (default),
%    Y = C.'\X if TRANS = 'T', and Y = C'\X if TRANS = 'C'.
%
%    See also MF2, MF3, MF_CHOLMV, MF_MV, MF_SV, MFX.

function Y = mf_cholsv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  assert(F.symm == 'p','FLAM:mf_cholsv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(mf_cholsv(F,conj(X),'c')); return; end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward/downward sweep
  if trans == 'n'
    for i = 1:n
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      Y(rd,:) = f.L\Y(rd,:);
      Y(sk,:) = Y(sk,:) - f.E*Y(rd,:);
    end
  else
    for i = n:-1:1
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      Y(rd,:) = Y(rd,:) - f.E'*Y(sk,:);
      Y(rd,:) = f.L'\Y(rd,:);
    end
  end
end