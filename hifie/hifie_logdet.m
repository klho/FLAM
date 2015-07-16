% HIFIE_LOGDET   Compute log-determinant using hierarchical interpolative
%                factorization for integral equations.
%
%    LD = HIFIE_LOGDET(F) produces the log-determinant LD of the factored matrix
%    F with 0 <= IMAG(LD) < 2*PI.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X.

function ld = hifie_logdet(F)
  ld = rskelf_logdet(F);
end