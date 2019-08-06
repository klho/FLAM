% RSKELF_LOGDET  Compute log-determinant using recursive skeletonization
%                factorization.
%
%    Typical complexity: O(N) in all dimensions.
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
    if     strcmpi(F.symm,'p'), ld = ld + 2*sum(log(diag(F.factors(i).L)));
    elseif strcmpi(F.symm,'h'), ld = ld + sum(log(F.factors(i).U));
    else
      ld = ld + sum(log(diag(F.factors(i).U))) + log(detperm(F.factors(i).p));
    end
  end

  % wrap imaginary part to [0, 2*PI)
  ld = real(ld) + 1i*mod(imag(ld),2*pi);
end