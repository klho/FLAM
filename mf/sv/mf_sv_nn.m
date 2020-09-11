% MF_SV_NN  Dispatch for MF_SV with F.SYMM = 'N' and TRANS = 'N'.

function X = mf_sv_nn(F,X)

  % initialize
  n = F.lvp(end);

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = F.factors(i).L\X(rd(F.factors(i).p),:);
    X(sk,:) = X(sk,:) - F.factors(i).E*X(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    X(rd,:) = X(rd,:) - F.factors(i).F*X(sk,:);
    X(rd,:) = F.factors(i).U\X(rd,:);
  end
end