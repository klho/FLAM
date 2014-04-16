% HIFIE_MV      Multiply using hierarchical interpolative factorization for
%               integral operators.
%
%    Y = HIFIE_MV(F,X) produces the matrix Y by applying the factored matrix F
%    to the matrix X.
%
%    Y = HIFIE_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_SV.

function Y = hifie_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  trans = lower(trans);
  if ~(strcmp(trans,'n') || strcmp(trans,'t') || strcmp(trans,'c'))
    error('FLAM:hifie_mv:invalidTrans', ...
          'Transpose parameter must be one of ''N'', ''T'', or ''C''.')
  end

  % handle transpose by conjugation
  if strcmp(trans,'t')
    Y = conj(hifie_mv(F,conj(X),'c'));
    return
  end

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    if strcmp(F.symm,'n')
      T = F.factors(i).T;
      if strcmp(trans,'n')
        G = F.factors(i).F;
      elseif strcmp(trans,'c')
        G = F.factors(i).E';
      end
    elseif strcmp(F.symm,'s')
      if strcmp(trans,'n')
        T = F.factors(i).T;
        G = F.factors(i).E.';
      elseif strcmp(trans,'c')
        T = conj(F.factors(i).T);
        G = F.factors(i).E';
      end
    elseif strcmp(F.symm,'h')
      T = F.factors(i).T;
      G = F.factors(i).E';
    end
    Y(sk,:) = Y(sk,:) + T*Y(rd,:);
    Y(rd,:) = Y(rd,:) + G*Y(sk,:);

    % apply diagonal blocks
    P = F.factors(i).P;
    L = F.factors(i).L;
    if strcmp(F.symm,'n') || strcmp(F.symm,'s')
      U = F.factors(i).U;
      if strcmp(trans,'n')
        Y(rd,:) = P'*(L*(U*Y(rd,:)));
      elseif strcmp(trans,'c')
        Y(rd,:) = U'*(L'*(P*Y(rd,:)));
      end
    elseif strcmp(F.symm,'h')
      D = F.factors(i).U;
      if strcmp(trans,'n')
        Y(rd,:) = P*(L*(D*(L'*(P'*Y(rd,:)))));
      elseif strcmp(trans,'c')
        Y(rd,:) = P*(L*(D'*(L'*(P'*Y(rd,:)))));
      end
    end
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    if strcmp(F.symm,'n')
      T = F.factors(i).T';
      if strcmp(trans,'n')
        E = F.factors(i).E;
      elseif strcmp(trans,'c')
        E = F.factors(i).F';
      end
    elseif strcmp(F.symm,'s')
      if strcmp(trans,'n')
        T = F.factors(i).T.';
        E = F.factors(i).E;
      elseif strcmp(trans,'c')
        T = F.factors(i).T';
        E = conj(F.factors(i).E);
      end
    elseif strcmp(F.symm,'h')
      T = F.factors(i).T';
      E = F.factors(i).E;
    end
    Y(sk,:) = Y(sk,:) + E*Y(rd,:);
    Y(rd,:) = Y(rd,:) + T*Y(sk,:);
  end
end