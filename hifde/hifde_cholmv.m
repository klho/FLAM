% HIFDE_CHOLMV  Multiply using generalized Cholesky factor from hierarchical
%               interpolative factorization for differential equations.
%
%    Typical complexity: about half that of HIFDE_MV.
%
%    Y = HIFDE_CHOLMV(F,X) produces the matrix Y by applying the generalized
%    Cholesky factor C of the factored matrix F = C*C' to the matrix X. Requires
%    that F be computed with the Hermitian positive-definite option.
%
%    Y = HIFDE_CHOLMV(F,X,TRANS) computes Y = C*X if TRANS = 'N' (default),
%    Y = C.'*X if TRANS = 'T', and Y = C'*X if TRANS = 'C'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_CHOLSV, HIFDE_MV,
%    HIFDE_SV.

function Y = hifde_cholmv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  assert(F.symm == 'p','FLAM:hifde_cholmv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(hifde_cholmv(F,conj(X),'c')); return; end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward/downward sweep
  if trans == 'n'
    for i = n:-1:1
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      T = F.factors(i).T;
      Y(sk,:) = Y(sk,:) + F.factors(i).E*Y(rd,:);
      Y(rd,:) = F.factors(i).L*Y(rd,:);
      if ~isempty(T), Y(rd,:) = Y(rd,:) + T'*Y(sk,:); end
    end
  else
    for i = 1:n
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      T = F.factors(i).T;
      if ~isempty(T), Y(sk,:) = Y(sk,:) + T*Y(rd,:); end
      Y(rd,:) = F.factors(i).L'*Y(rd,:);
      Y(rd,:) = Y(rd,:) + F.factors(i).E'*Y(sk,:);
    end
  end
end