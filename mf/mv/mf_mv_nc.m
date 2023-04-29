% MF_MV_NC  Dispatch for MF_MV with F.SYMM = 'N' and TRANS = 'C'.

function X = mf_mv_nc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.L'*X(rd(f.p),:);
    X(rd,:) = X(rd,:) + f.E'*X(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(sk,:) = X(sk,:) + f.F'*X(rd,:);
    X(rd,:) = f.U'*X(rd,:);
  end
end