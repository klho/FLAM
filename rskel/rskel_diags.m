% RSKEL_DIAGS   Extract diagonal of a square matrix with identical row and
%               column correspondences using recursive skeletonization.
%
%    D = RSKEL_DIAGS(F) produces the diagonal D of the compressed matrix F.
%
%    See also RSKEL.

function D = rskel_diags(F)

  % check inputs
  assert(F.M == F.N,'FLAM:rskel_diags:invalidDim','Matrix must be square.')

  % extract diagonal
  N = F.N;
  D = zeros(N,1);
  for i = 1:F.lvpd(end)
    j = F.D(i).i;
    k = F.D(i).j;
    E = F.D(i).D;
    [j,k] = ndgrid(F.D(i).i,F.D(i).j);
    idx = j == k;
    D(j(idx)) = E(idx);
  end
end