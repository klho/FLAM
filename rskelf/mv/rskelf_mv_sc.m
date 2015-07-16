% RSKELF_MV_SC   Dispatch for RSKELF_MV with F.SYMM = 'S' and TRANS = 'C'.
%
%    See also RSKELF, RSKELF_MV.

function Y = rskelf_mv_sc(F,X)

  % initialize
  n = F.lvp(end);
  Y = X;

  % upward sweep
  for i = 1:n
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(sk,:) = Y(sk,:) + conj(F.factors(i).T)*Y(rd,:);
    Y(rd,:) = F.factors(i).L'*Y(rd,:);
    Y(rd,:) = Y(rd,:) + F.factors(i).E'*Y(sk,:);
  end

  % downward sweep
  for i = n:-1:1
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    Y(sk,:) = Y(sk,:) + F.factors(i).F'*Y(rd,:);
    Y(rd,:) = F.factors(i).U'*Y(rd,:);
    Y(rd,:) = Y(rd,:) + F.factors(i).T'*Y(sk,:);
  end
end