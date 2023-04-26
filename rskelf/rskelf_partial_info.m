% RSKELF_PARTIAL_INFO  Retrieve compressed skeleton information from partial
%                      recursive skeletonization factorization.
%
%    [SK,S] = RSKELF_PARTIAL_INFO(F) returns the skeleton indices SK remaining
%    at the end of the factorization as well as the accumulated matrix updates
%    assembled as a sparse matrix S. In other words, if A denotes the original
%    matrix, then the remaining skeleton submatrix after compression by the
%    partial factorization is A(SK,SK) + S.
%
%    See also RSKELF, RSKELF_PARTIAL_MV, RSKELF_PARTIAL_SV.

function [sk,S] = rskelf_partial_info(F)
  if isfield(F,'Si'), sk = F.Si;
  else,               sk = [];
  end
  if isfield(F,'S'), S = F.S;
  else,              S = sparse(0,0);
end