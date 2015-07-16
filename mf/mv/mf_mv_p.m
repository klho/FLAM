% MF_MV_P  Dispatch for MF_MV with F.SYMM = 'P'.
%
%    See also MF2, MF3, MF_MV, MFX.

function Y = mf_mv_p(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(rd,:) = F.factors(i).L'*Y(rd,:);
    Y(rd,:) = Y(rd,:) + F.factors(i).E'*Y(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(sk,:) = Y(sk,:) + F.factors(i).E*Y(rd,:);
    Y(rd,:) = F.factors(i).L*Y(rd,:);
  end
end