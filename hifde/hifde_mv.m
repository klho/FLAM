% HIFDE_MV      Multiply using hierarchical interpolative factorization for
%               differential operators.
%
%    Y = HIFDE_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = HIFDE_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_SV.

function Y = hifde_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifde_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(hifde_mv(F,conj(X),'c'));
    return
  end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    if strcmpi(F.symm,'n')
      if strcmpi(trans,'n')
        G = F.factors(i).F;
      else
        G = F.factors(i).E';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        G = F.factors(i).E.';
      else
        T = conj(T);
        G = F.factors(i).E';
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      G = F.factors(i).E';
    end
    if ~isempty(T)
      Y(sk,:) = Y(sk,:) + T*Y(rd,:);
    end
    Y(rd,:) = Y(rd,:) + G*Y(sk,:);

    % apply diagonal blocks
    L = F.factors(i).L;
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      U = F.factors(i).U;
      if strcmpi(trans,'n')
        Y(rd,:) = L*(U*Y(rd,:));
      else
        Y(rd,:) = U'*(L'*Y(rd,:));
      end
    elseif strcmpi(F.symm,'h')
      if strcmpi(trans,'n')
        D = F.factors(i).U;
      else
        D = F.factors(i).U';
      end
      Y(rd,:) = L*(D*(L'*Y(rd,:)));
    elseif strcmpi(F.symm,'p')
      Y(rd,:) = L*(L'*Y(rd,:));
    end
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T';
    if strcmpi(F.symm,'n')
      if strcmpi(trans,'n')
        E = F.factors(i).E;
      else
        E = F.factors(i).F';
      end
    elseif strcmpi(F.symm,'s')
      if strcmpi(trans,'n')
        T = conj(T);
        E = F.factors(i).E;
      else
        E = conj(F.factors(i).E);
      end
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      E = F.factors(i).E;
    end
    Y(sk,:) = Y(sk,:) + E*Y(rd,:);
    if ~isempty(T)
      Y(rd,:) = Y(rd,:) + T*Y(sk,:);
    end
  end
end