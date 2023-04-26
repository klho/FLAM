% RSKELF_CHOLMV  Multiply using generalized Cholesky factor from recursive
%                skeletonization factorization.
%
%    Typical complexity: about half that of RSKELF_MV.
%
%    Y = RSKELF_CHOLMV(F,X) produces the matrix Y by applying the generalized
%    Cholesky factor C of the factored matrix F = C*C' to the matrix X. Requires
%    that F be computed with the Hermitian positive definite option.
%
%    Y = RSKELF_CHOLMV(F,X,TRANS) computes Y = C*X if TRANS = 'N' (default),
%    Y = C.'*X if TRANS = 'T', and Y = C'*X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLSV, RSKELF_MV, RSKELF_SV.

function Y = rskelf_cholmv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  assert(F.symm == 'p','FLAM:rskelf_cholmv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(rskelf_cholmv(F,conj(X),'c')); return; end

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
      Y(rd,:) = Y(rd,:) + f.T'*Y(sk,:);
    end
  else
    for i = 1:n
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      Y(sk,:) = Y(sk,:) + f.T*Y(rd,:);
      Y(rd,:) = f.L'*Y(rd,:);
      Y(rd,:) = Y(rd,:) + f.E'*Y(sk,:);
    end
  end
end