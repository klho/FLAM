% RSKELF_MV  Multiply using recursive skeletonization factorization.
%
%    Typical complexity: O(N) in 1D and O(N^(2*(1 - 1/D))) in D dimensions.
%
%    Y = RSKELF_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = RSKELF_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLMV, RSKELF_CHOLSV, RSKELF_SV.

function X = rskelf_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', X = conj(rskelf_mv(F,conj(X),'c')); return; end

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if trans == 'n', X = rskelf_mv_nn(F,X,3);
    else,            X = rskelf_mv_nc(F,X,3);
    end
  elseif F.symm == 's'
    if trans == 'n', X = rskelf_mv_sn(F,X,3);
    else,            X = rskelf_mv_sc(F,X,3);
    end
  elseif F.symm == 'h', X = rskelf_mv_h(F,X,3);
  elseif F.symm == 'p', X = rskelf_mv_p(F,X,3);
  end
end