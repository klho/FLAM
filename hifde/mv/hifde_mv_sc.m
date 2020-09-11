% HIFDE_MV_SC  Dispatch for HIFDE_MV with F.SYMM = 'S' and TRANS = 'C'.

function X = hifde_mv_sc(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    if ~isempty(T), X(sk,:) = X(sk,:) + conj(T)*X(rd,:); end
    X(rd,:) = F.factors(i).L'*X(rd(F.factors(i).p),:);
    X(rd,:) = X(rd,:) + F.factors(i).E'*X(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    T = F.factors(i).T;
    X(sk,:) = X(sk,:) + F.factors(i).F'*X(rd,:);
    X(rd,:) = F.factors(i).U'*X(rd,:);
    if ~isempty(T), X(rd,:) = X(rd,:) + T'*X(sk,:); end
  end
end