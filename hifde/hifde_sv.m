% HIFDE_SV  Solve using hierarchical interpolative factorization for
%           differential equations.
%
%    Typical complexity: same as HIFDE_MV.
%
%    Y = HIFDE_SV(F,X) produces the matrix Y by applying the inverse of the
%    factored matrix F to the matrix X.
%
%    Y = HIFDE_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_CHOLMV, HIFDE_CHOLSV,
%    HIFDE_MV.

function X = hifde_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', X = conj(hifde_sv(F,conj(X),'c')); return; end

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if trans == 'n', X = hifde_sv_nn(F,X);
    else,            X = hifde_sv_nc(F,X);
    end
  elseif F.symm == 's'
    if trans == 'n', X = hifde_sv_sn(F,X);
    else,            X = hifde_sv_sc(F,X);
    end
  elseif F.symm == 'h', X = hifde_sv_h(F,X);
  elseif F.symm == 'p', X = hifde_sv_p(F,X);
  end
end