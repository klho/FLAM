% RSKELF_CHOLMV  Multiply using generalized Cholesky factor from recursive
%                skeletonization factorization.
%
%    Y = RSKELF_CHOLMV(F,X) produces the matrix Y by applying the generalized
%    Cholesky factor C of the factored matrix F = C*C' to the matrix X. Requires
%    that F be computed with the Hermitian positive-definite option.
%
%    Y = RSKELF_CHOLMV(F,X,TRANS) computes Y = C*X if TRANS = 'N' (default),
%    Y = C.'*X if TRANS = 'T', and Y = C'*X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLSV, RSKELF_MV, RSKELF_SV.

function Y = rskelf_cholmv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(F.symm,'p'),'FLAM:rskelf_cholmv:invalidSymm', ...
         'Symmetry parameter must be ''P''.')
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskelf_cholmv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskelf_cholmv(F,conj(X),'c'));
    return
  end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward/downward sweep
  if strcmpi(trans,'n')
    for i = n:-1:1
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      Y(sk,:) = Y(sk,:) + F.factors(i).E*Y(rd,:);
      Y(rd,:) = F.factors(i).L*Y(rd,:);
      Y(rd,:) = Y(rd,:) + F.factors(i).T'*Y(sk,:);
    end
  else
    for i = 1:n
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      Y(sk,:) = Y(sk,:) + F.factors(i).T*Y(rd,:);
      Y(rd,:) = F.factors(i).L'*Y(rd,:);
      Y(rd,:) = Y(rd,:) + F.factors(i).E'*Y(sk,:);
    end
  end
end