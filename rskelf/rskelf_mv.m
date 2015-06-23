% RSKELF_MV     Multiply using recursive skeletonization factorization.
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
    Y(sk,:) = Y(sk,:) + T*Y(rd,:);
    Y(rd,:) = U*Y(rd,:);
    Y(rd,:) = Y(rd,:) + G*Y(sk,:);
    if strcmpi(F.symm,'h')
      Y(rd,:) = F.factors(i).U*Y(rd,:);
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
    Y(sk,:) = Y(sk,:) + E*Y(rd,:);
    Y(rd,:) = L*Y(rd,:);
    Y(rd,:) = Y(rd,:) + T*Y(sk,:);
  end
end