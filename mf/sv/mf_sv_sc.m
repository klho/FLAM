% MF_SV_SC   Dispatch for MF_SV with F.SYMM = 'S' and TRANS = 'C'.
%
%    See also MF2, MF3, MF_MV, MFX.

function Y = mf_sv_sc(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    L = F.factors(i).U';
    Y(rd,:) = L\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).F'*Y(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(rd,:) = Y(rd,:) - F.factors(i).E'*Y(sk,:);
    U = F.factors(i).L';
    Y(rd,:) = U\Y(rd,:);
  end
end