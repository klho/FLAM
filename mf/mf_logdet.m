% MF_LOGDET  Compute log-determinant using multifrontal factorization.
%
%    LD = MF_LOGDET(F) produces the log-determinant LD of the factored matrix F
%    with 0 <= IMAG(LD) < 2*PI.
%
%    See also MF2, MF3, MFX.

function ld = mf_logdet(F)
  ld = rskelf_logdet(F);
end