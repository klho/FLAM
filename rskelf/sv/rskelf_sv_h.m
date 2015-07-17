% RSKELF_SV_H  Dispatch for RSKELF_SV with F.SYMM = 'H'.
%
%    See also RSKELF, RSKELF_SV.

function Y = rskelf_sv_h(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(rd,:) = Y(rd,:) - F.factors(i).T'*Y(sk,:);
    Y(rd,:) = F.factors(i).L\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).E*Y(rd,:);
    Y(rd,:) = F.factors(i).U\Y(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    U = F.factors(i).L';
    Y(rd,:) = Y(rd,:) - F.factors(i).E'*Y(sk,:);
    Y(rd,:) = U\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).T*Y(rd,:);
  end
end