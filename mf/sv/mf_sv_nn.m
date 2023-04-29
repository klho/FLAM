% MF_SV_NN  Dispatch for MF_SV with F.SYMM = 'N' and TRANS = 'N'.

function X = mf_sv_nn(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.L\X(rd(f.p),:);
    X(sk,:) = X(sk,:) - f.E*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = X(rd,:) - f.F*X(sk,:);
    X(rd,:) = f.U\X(rd,:);
  end
end