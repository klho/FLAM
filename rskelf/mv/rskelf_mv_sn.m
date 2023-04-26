% RSKELF_MV_SN  Dispatch for RSKELF_MV with F.SYMM = 'S' and TRANS = 'N'.

function X = rskelf_mv_sn(F,X,mode)

  % initialize
  n = F.lvp(end);

  % upward sweep
  if bitget(mode,1)
    for i = 1:n
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      X(sk,:) = X(sk,:) + f.T*X(rd,:);
      X(rd,:) = f.U*X(rd,:);
      X(rd,:) = X(rd,:) + f.F*X(sk,:);
    end
  end

  % downward sweep
  if bitget(mode,2)
    for i = n:-1:1
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      X(sk,:) = X(sk,:) + f.E*X(rd,:);
      X(rd(f.p),:) = f.L*X(rd,:);
      X(rd,:) = X(rd,:) + f.T.'*X(sk,:);
    end
  end
end