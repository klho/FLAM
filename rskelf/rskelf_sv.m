% RSKELF_SV  Solve using recursive skeletonization factorization.
%
%    Y = RSKELF_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = RSKELF_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also RSKELF, RSKELF_CHOLMV, RSKELF_CHOLSV, RSKELF_MV.

function Y = rskelf_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskelf_sv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskelf_sv(F,conj(X),'c'));
    return
  end

  % dispatch
  if strcmpi(F.symm,'n')
    if strcmpi(trans,'n')
      Y = rskelf_sv_nn(F,X);
    else
      Y = rskelf_sv_nc(F,X);
    end
  elseif strcmpi(F.symm,'s')
    if strcmpi(trans,'n')
      Y = rskelf_sv_sn(F,X);
    else
      Y = rskelf_sv_sc(F,X);
    end
  elseif strcmpi(F.symm,'h')
    Y = rskelf_sv_h(F,X);
  elseif strcmpi(F.symm,'p')
    Y = rskelf_sv_p(F,X);
  end
end