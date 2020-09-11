% MF_SV_P  Dispatch for MF_SV with F.SYMM = 'P'.

function X = mf_sv_p(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = F.factors(i).L\X(rd,:);
    X(sk,:) = X(sk,:) - F.factors(i).E*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = X(rd,:) - F.factors(i).E'*X(sk,:);
    X(rd,:) = F.factors(i).L'\X(rd,:);
  end
end