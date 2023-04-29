% MF_MV_NN  Dispatch for MF_MV with F.SYMM = 'N' and TRANS = 'N'.

function X = mf_mv_nn(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.U*X(rd,:);
    X(rd,:) = X(rd,:) + f.F*X(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(sk,:) = X(sk,:) + f.E*X(rd,:);
    X(rd(f.p),:) = f.L*X(rd,:);
  end
end