% RSKELF_MV  Multiply using recursive skeletonization factorization.
%
%    Y = RSKELF_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = RSKELF_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLMV, RSKEL_CHOLSV, RSKELF_SV.

function Y = rskelf_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskelf_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskelf_mv(F,conj(X),'c'));
    return
  end

  % dispatch
  if strcmpi(F.symm,'n')
    if strcmpi(trans,'n')
      Y = rskelf_mv_nn(F,X);
    else
      Y = rskelf_mv_nc(F,X);
    end
  elseif strcmpi(F.symm,'s')
    if strcmpi(trans,'n')
      Y = rskelf_mv_sn(F,X);
    else
      Y = rskelf_mv_sc(F,X);
    end
  elseif strcmpi(F.symm,'h')
    Y = rskelf_mv_h(F,X);
  elseif strcmpi(F.symm,'p')
    Y = rskelf_mv_p(F,X);
  end
end