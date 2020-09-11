% HIFDE_SV_SC  Dispatch for HIFDE_SV with F.SYMM = 'S' and TRANS = 'C'.

function X = hifde_sv_sc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    if ~isempty(T), X(rd,:) = X(rd,:) - T'*X(sk,:); end
    X(rd,:) = F.factors(i).U'\X(rd,:);
    X(sk,:) = X(sk,:) - F.factors(i).F'*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    X(rd,:) = X(rd,:) - F.factors(i).E'*X(sk,:);
    X(rd(F.factors(i).p),:) = F.factors(i).L'\X(rd,:);
    if ~isempty(T), X(sk,:) = X(sk,:) - conj(T)*X(rd,:); end
  end
end