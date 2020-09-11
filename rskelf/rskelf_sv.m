% RSKELF_SV  Solve using recursive skeletonization factorization.
%
%    Typical complexity: same as RSKELF_MV.
%
%    Y = RSKELF_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = RSKELF_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLMV, RSKELF_CHOLSV, RSKELF_MV.

function X = rskelf_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', X = conj(rskelf_sv(F,conj(X),'c')); return; end

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if trans == 'n', X = rskelf_sv_nn(F,X);
    else,            X = rskelf_sv_nc(F,X);
    end
  elseif F.symm == 's'
    if trans == 'n', X = rskelf_sv_sn(F,X);
    else,            X = rskelf_sv_sc(F,X);
    end
  elseif F.symm == 'h', X = rskelf_sv_h(F,X);
  elseif F.symm == 'p', X = rskelf_sv_p(F,X);
  end
end