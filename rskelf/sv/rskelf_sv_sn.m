% RSKELF_SV_SN   Dispatch for RSKELF_SV with F.SYMM = 'S' and TRANS = 'N'.
%
%    See also RSKELF, RSKELF_SV.

function Y = rskelf_sv_sn(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(rd,:) = Y(rd,:) - F.factors(i).T.'*Y(sk,:);
    Y(rd,:) = F.factors(i).L\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).E*Y(rd,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(rd,:) = Y(rd,:) - F.factors(i).F*Y(sk,:);
    Y(rd,:) = F.factors(i).U\Y(rd,:);
    Y(sk,:) = Y(sk,:) - F.factors(i).T*Y(rd,:);
  end
end