% HIFDE_SV      Solve using hierarchical interpolative factorization for
%               differential operators.
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
  if ~(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'))
    error('FLAM:hifde_sv:invalidTrans', ...
          'Transpose parameter must be one of ''N'', ''T'', or ''C''.')
  end

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
    if strcmpi(F.symm,'n')
      T = F.factors(i).T';
      if strcmpi(trans,'n')
        E = F.factors(i).E;
      else
        E = F.factors(i).F';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        T = F.factors(i).T.';
        E = F.factors(i).E;
      else
        T = F.factors(i).T';
        E = conj(F.factors(i).E);
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      T = F.factors(i).T';
      E = F.factors(i).E;
    end
    if ~isempty(T)
      Y(rd,:) = Y(rd,:) - T*Y(sk,:);
    end
    Y(sk,:) = Y(sk,:) - E*Y(rd,:);

    % apply diagonal blocks
    L = F.factors(i).L;
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      U = F.factors(i).U;
      if strcmpi(trans,'n')
        Y(rd,:) = U\(L\Y(rd,:));
      else
        Y(rd,:) = L'\(U'\Y(rd,:));
      end
    elseif strcmpi(F.symm,'h')
      if strcmpi(trans,'n')
        D = F.factors(i).U;
      else
        D = F.factors(i).U';
      end
      Y(rd,:) = L'\(D\(L\Y(rd,:)));
    elseif strcmpi(F.symm,'p')
      Y(rd,:) = L'\(L\Y(rd,:));
    end
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    if strcmpi(F.symm,'n')
      T = F.factors(i).T;
      if strcmpi(trans,'n')
        G = F.factors(i).F;
      else
        G = F.factors(i).E';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        T = F.factors(i).T;
        G = F.factors(i).E.';
      else
        T = conj(F.factors(i).T);
        G = F.factors(i).E';
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      T = F.factors(i).T;
      G = F.factors(i).E';
    end
    Y(rd,:) = Y(rd,:) - G*Y(sk,:);
    if ~isempty(T)
      Y(sk,:) = Y(sk,:) - T*Y(rd,:);
    end
  end
end