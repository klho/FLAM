% MF_MV  Multiply using multifrontal factorization.
%
%    Y = MF_MV(F,X) produces the matrix Y by applying the factored matrix F to
%    the matrix X.
%
%    Y = MF_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default), Y = F.'*X
%    if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also MF2, MF3, MF_CHOLMV, MF_CHOLSV, MF_SV, MFX.

function Y = mf_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:mf_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(mf_mv(F,conj(X),'c'));
    return
  end

  % dispatch
  if strcmpi(F.symm,'n')
    if strcmpi(trans,'n')
      Y = mf_mv_nn(F,X);
    else
      Y = mf_mv_nc(F,X);
    end
  elseif strcmpi(F.symm,'s')
    if strcmpi(trans,'n')
      Y = mf_mv_sn(F,X);
    else
      Y = mf_mv_sc(F,X);
    end
  elseif strcmpi(F.symm,'h')
    Y = mf_mv_h(F,X);
  elseif strcmpi(F.symm,'p')
    Y = mf_mv_p(F,X);
  end
end