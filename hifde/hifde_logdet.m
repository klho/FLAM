% HIFDE_LOGDET  Compute log-determinant using hierarchical interpolative
%               factorization for differential equations.
%
%    Typical complexity: O(N) in all dimensions.
%
%    LD = HIFDE_LOGDET(F) produces the log-determinant LD of the factored matrix
%    F with 0 <= IMAG(LD) < 2*PI.
%
%    See also HIFDE2, HIFDE2X, HIFDE3, HIFDE3X.

function ld = hifde_logdet(F)
  ld = rskelf_logdet(F);
end