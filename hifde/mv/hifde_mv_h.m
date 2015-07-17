% HIFDE_MV_H   Dispatch for HIFDE_MV with F.SYMM = 'H'.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X, HIFDE_MV.

function Y = hifde_mv_h(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    if ~isempty(T)
      Y(sk,:) = Y(sk,:) + T*Y(rd,:);
    end
    Y(rd,:) = F.factors(i).L'*Y(rd,:);
    Y(rd,:) = Y(rd,:) + F.factors(i).E'*Y(sk,:);
    Y(rd,:) = F.factors(i).U*Y(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    Y(sk,:) = Y(sk,:) + F.factors(i).E*Y(rd,:);
    Y(rd,:) = F.factors(i).L*Y(rd,:);
    if ~isempty(T)
      Y(rd,:) = Y(rd,:) + T'*Y(sk,:);
    end
  end
end