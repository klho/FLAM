% MF_MV_P  Dispatch for MF_MV with F.SYMM = 'P'.

function X = mf_mv_p(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.L'*X(rd,:);
    X(rd,:) = X(rd,:) + f.E'*X(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(sk,:) = X(sk,:) + f.E*X(rd,:);
    X(rd,:) = f.L*X(rd,:);
  end
end