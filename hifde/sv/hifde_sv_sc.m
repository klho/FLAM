% HIFDE_SV_SC  Dispatch for HIFDE_SV with F.SYMM = 'S' and TRANS = 'C'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_MV.

function Y = hifde_sv_sc(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    L = F.factors(i).U';
    if ~isempty(T)
      Y(rd,:) = Y(rd,:) - T'*Y(sk,:);
    end
    Y(rd,:) = L\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).F'*Y(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    U = F.factors(i).L';
    Y(rd,:) = Y(rd,:) - F.factors(i).E'*Y(sk,:);
    Y(rd,:) = U\Y(rd,:);
    if ~isempty(T)
      Y(sk,:) = Y(sk,:) - conj(F.factors(i).T)*Y(rd,:);
    end
  end
end