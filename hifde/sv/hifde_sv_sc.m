% HIFDE_SV_SC  Dispatch for HIFDE_SV with F.SYMM = 'S' and TRANS = 'C'.

function X = hifde_sv_sc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    if ~isempty(f.T), X(rd,:) = X(rd,:) - f.T'*X(sk,:); end
    X(rd,:) = f.U'\X(rd,:);
    X(sk,:) = X(sk,:) - f.F'*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(rd,:) = X(rd,:) - f.E'*X(sk,:);
    X(rd(f.p),:) = f.L'\X(rd,:);
    if ~isempty(f.T), X(sk,:) = X(sk,:) - conj(f.T)*X(rd,:); end
  end
end