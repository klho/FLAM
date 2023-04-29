% MF_SV_NC  Dispatch for MF_SV with F.SYMM = 'N' and TRANS = 'C'.

function X = mf_sv_nc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = f.U'\X(rd,:);
    X(sk,:) = X(sk,:) - f.F'*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = X(rd,:) - f.E'*X(sk,:);
    X(rd(f.p),:) = f.L'\X(rd,:);
  end
end