% HIFIE_ID  Compression function for HIFIE2 and HIFIE3.

function [sk,rd,T] = hifie_id(K,K1,K2,rank_or_tol)
  [sk,rd,T] = id(K,rank_or_tol);
end