% RSKELF_SV_SN  Dispatch for RSKELF_SV with F.SYMM = 'S' and TRANS = 'N'.

function X = rskelf_sv_sn(F,X,mode)

  % initialize
  n = F.lvp(end);

  % upward sweep
  if bitget(mode,1)
    for i = 1:n
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      X(rd,:) = X(rd,:) - f.T.'*X(sk,:);
      X(rd,:) = f.L\X(rd(f.p),:);
      X(sk,:) = X(sk,:) - f.E*X(rd,:);
    end
  end

  % downward sweep
  if bitget(mode,2)
    for i = n:-1:1
      f = F.factors(i);
      sk = f.sk; rd = f.rd;
      X(rd,:) = X(rd,:) - f.F*X(sk,:);
      X(rd,:) = f.U\X(rd,:);
      X(sk,:) = X(sk,:) - f.T*X(rd,:);
    end
  end
end