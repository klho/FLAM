% RSKELF_LOGDET  Compute log-determinant using recursive skeletonization
%                factorization.
%
%    LD = RSKELF_LOGDET(F) produces the log-determinant LD of the factored
%    matrix F with 0 <= IMAG(LD) < 2*PI.
%
%    See also RSKELF.

function ld = rskelf_logdet(F)

  % initialize
  n = F.lvp(end);
  ld = 0;

  % loop over nodes
  for i = 1:n
    L = F.factors(i).L;
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      ld = ld + log(det(L)) + sum(log(diag(F.factors(i).U)));
    elseif strcmpi(F.symm,'h')
      ld = ld + sum(log(diag(F.factors(i).U)));
    elseif strcmpi(F.symm,'p')
      ld = ld + 2*sum(log(diag(L)));
    end
  end

  % wrap imaginary part to [0, 2*PI)
  ld = real(ld) + 1i*mod(imag(ld),2*pi);
end