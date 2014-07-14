% MF_MV Multiply using multifrontal factorization.
%
%    Y = MF_MV(F,X) produces the matrix Y by applying the factored matrix F to
%    the matrix X.
%
%    Y = MF_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default), Y = F.'*X
%    if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also MF2, MF3, MF_SV, MFX.

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

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    if strcmpi(F.symm,'n')
      if strcmpi(trans,'n')
        G = F.factors(i).F;
        U = F.factors(i).U;
      else
        G = F.factors(i).E';
        U = F.factors(i).L';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        G = F.factors(i).F;
        U = F.factors(i).U;
      else
        G = F.factors(i).E';
        U = F.factors(i).L';
      end
    elseif strcmpi(F.symm,'h')
      if strcmpi(trans,'n')
        G = F.factors(i).F;
        U = F.factors(i).U*F.factors(i).L';
      else
        G = F.factors(i).E';
        U = F.factors(i).L';
      end
    elseif strcmpi(F.symm,'p')
      G = F.factors(i).E';
      U = F.factors(i).L';
    end
    Y(rd,:) = U*Y(rd,:);
    Y(rd,:) = Y(rd,:) + G*Y(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    if strcmpi(F.symm,'n')
      if strcmpi(trans,'n')
        E = F.factors(i).E;
        L = F.factors(i).L;
      else
        E = F.factors(i).F';
        L = F.factors(i).U';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        E = F.factors(i).E;
        L = F.factors(i).L;
      else
        E = F.factors(i).F';
        L = F.factors(i).U';
      end
    elseif strcmpi(F.symm,'h')
      if strcmpi(trans,'n')
        E = F.factors(i).E;
        L = F.factors(i).L;
      else
        E = F.factors(i).F';
        L = F.factors(i).L*F.factors(i).U;
      end
    elseif strcmpi(F.symm,'p')
      E = F.factors(i).E;
      L = F.factors(i).L;
    end
    Y(sk,:) = Y(sk,:) + E*Y(rd,:);
    Y(rd,:) = L*Y(rd,:);
  end
end