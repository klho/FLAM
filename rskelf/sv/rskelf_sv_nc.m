% RSKELF_SV_NC  Dispatch for RSKELF_SV with F.SYMM = 'N' and TRANS = 'C'.

function X = rskelf_sv_nc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = X(rd,:) - F.factors(i).T'*X(sk,:);
    X(rd,:) = F.factors(i).U'\X(rd,:);
    X(sk,:) = X(sk,:) - F.factors(i).F'*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = X(rd,:) - F.factors(i).E'*X(sk,:);
    X(rd(F.factors(i).p),:) = F.factors(i).L'\X(rd,:);
    X(sk,:) = X(sk,:) - F.factors(i).T*X(rd,:);
  end
end