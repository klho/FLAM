% RSKELF_SV_NN  Dispatch for RSKELF_SV with F.SYMM = 'N' and TRANS = 'N'.

function X = rskelf_sv_nn(F,X,mode)

  % initialize
  n = F.lvp(end);

  % upward sweep
  if bitget(mode,1)
    for i = 1:n
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      X(rd,:) = X(rd,:) - F.factors(i).T'*X(sk,:);
      X(rd,:) = F.factors(i).L\X(rd(F.factors(i).p),:);
      X(sk,:) = X(sk,:) - F.factors(i).E*X(rd,:);
    end
  end

  % downward sweep
  if bitget(mode,2)
    for i = n:-1:1
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      X(rd,:) = X(rd,:) - F.factors(i).F*X(sk,:);
      X(rd,:) = F.factors(i).U\X(rd,:);
      X(sk,:) = X(sk,:) - F.factors(i).T*X(rd,:);
    end
  end
end