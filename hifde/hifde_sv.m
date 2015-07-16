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

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T';
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
        T = conj(T);
        E = F.factors(i).E;
        L = F.factors(i).L;
      else
        E = F.factors(i).F';
        L = F.factors(i).U';
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      E = F.factors(i).E;
      L = F.factors(i).L;
    end
    if ~isempty(T)
      Y(rd,:) = Y(rd,:) - T*Y(sk,:);
    end
    Y(rd,:) = L\Y(rd,:);
    Y(sk,:) = Y(sk,:) - E*Y(rd,:);
    if strcmpi(F.symm,'h')
      Y(rd,:) = F.factors(i).U\Y(rd,:);
    end
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
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
        T = conj(T);
        G = F.factors(i).E';
        U = F.factors(i).L';
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      G = F.factors(i).E';
      U = F.factors(i).L';
    end
    Y(rd,:) = Y(rd,:) - G*Y(sk,:);
    Y(rd,:) = U\Y(rd,:);
    if ~isempty(T)
      Y(sk,:) = Y(sk,:) - T*Y(rd,:);
    end
  end
end