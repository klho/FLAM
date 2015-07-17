% HIFDE_SV   Solve using hierarchical interpolative factorization for
%            differential equations.
%
%    Y = HIFDE_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = HIFDE_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_MV.

function Y = hifde_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifde_sv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(hifde_sv(F,conj(X),'c'));
    return
  end

  % dispatch
  if strcmpi(F.symm,'n')
    if strcmpi(trans,'n')
      Y = hifde_sv_nn(F,X);
    else
      Y = hifde_sv_nc(F,X);
    end
  elseif strcmpi(F.symm,'s')
    if strcmpi(trans,'n')
      Y = hifde_sv_sn(F,X);
    else
      Y = hifde_sv_sc(F,X);
    end
  elseif strcmpi(F.symm,'h')
    Y = hifde_sv_h(F,X);
  elseif strcmpi(F.symm,'p')
    Y = hifde_sv_p(F,X);
  end
end