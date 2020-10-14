% RSKELF_PARTIAL_MV  Multiply using partial recursive skeletonization
%                    factorization.
%
%    Like RSKELF_MV but using a user-supplied function of the form
%    Y = MVFUN(X,TRANS) to apply the compressed skeleton submatrix. More
%    specifically, denote by A the original matrix, and let SK and S be the
%    remaining skeletons and sparse matrix modifications as returned from
%    RSKELF_PARTIAL_INFO. Then Y = MVFUN(X,TRANS) computes
%
%      - Y = (A(SK,SK) + S)  *X if TRANS = 'N'
%      - Y = (A(SK,SK) + S).'*X if TRANS = 'T'
%      - Y = (A(SK,SK) + S) '*X if TRANS = 'C'
%
%    Y = RSKELF_PARTIAL_MV(F,X,MVFUN) produces the matrix Y by applying the
%    partially factored matrix F to the matrix X using the function MVFUN to
%    apply the compressed skeleton submatrix.
%
%    Y = RSKELF_PARTIAL_MV(F,X,MVFUN,TRANS) computes Y = F*X if TRANS = 'N'
%    (default), Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_MV, RSKELF_PARTIAL_INFO, RSKELF_PARTIAL_SV.

function X = rskelf_partial_mv(F,X,mvfun,trans)

  % set default parameters
  if nargin < 4 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', X = conj(rskelf_mv_partial(F,conj(X),'c')); return; end

  % upward sweep
  if F.symm == 'n'
    if trans == 'n', X = rskelf_mv_nn(F,X,1);
    else,            X = rskelf_mv_nc(F,X,1);
    end
  elseif F.symm == 's'
    if trans == 'n', X = rskelf_mv_sn(F,X,1);
    else,            X = rskelf_mv_sc(F,X,1);
    end
  elseif F.symm == 'h', X = rskelf_mv_h(F,X,1);
  elseif F.symm == 'p', X = rskelf_mv_p(F,X,1);
  end

  % apply skeleton matrix
  X(F.Si,:) = mvfun(X(F.Si,:),trans);

  % downward sweep
  if F.symm == 'n'
    if trans == 'n', X = rskelf_mv_nn(F,X,2);
    else,            X = rskelf_mv_nc(F,X,2);
    end
  elseif F.symm == 's'
    if trans == 'n', X = rskelf_mv_sn(F,X,2);
    else,            X = rskelf_mv_sc(F,X,2);
    end
  elseif F.symm == 'h', X = rskelf_mv_h(F,X,2);
  elseif F.symm == 'p', X = rskelf_mv_p(F,X,2);
  end
end