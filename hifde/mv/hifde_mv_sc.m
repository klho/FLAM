% HIFDE_MV_SC  Dispatch for HIFDE_MV with F.SYMM = 'S' and TRANS = 'C'.

function X = hifde_mv_sc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    if ~isempty(f.T), X(sk,:) = X(sk,:) + conj(f.T)*X(rd,:); end
    X(rd,:) = f.L'*X(rd(f.p),:);
    X(rd,:) = X(rd,:) + f.E'*X(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    f = F.factors(i);
    sk = f.sk; rd = f.rd;
    X(sk,:) = X(sk,:) + f.F'*X(rd,:);
    X(rd,:) = f.U'*X(rd,:);
    if ~isempty(f.T), X(rd,:) = X(rd,:) + f.T'*X(sk,:); end
  end
end