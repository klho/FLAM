% MF_SV_P  Dispatch for MF_SV with F.SYMM = 'P'.

function X = mf_sv_p(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.L\X(rd,:);
    X(sk,:) = X(sk,:) - f.E*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = X(rd,:) - f.E'*X(sk,:);
    X(rd,:) = f.L'\X(rd,:);
  end
end