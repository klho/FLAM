% SPPUSH3  Append to 3-array sparse matrix representation.
%
%   [I,J,V,NZ] = SPPUSH3(I,J,V,NZ,IA,JA,VA) overwrites I(NZ+(1:N)) = IA,
%   J(NZ+(1:N)) = JA, and V(NZ+(1:N)) = VA, where
%   N = LENGTH(IA) = LENGTH(JA) = LENGTH(VA), and replaces NZ := NZ + N,
%   appropriately reallocating and expanding the capacities of I, J, and V as
%   needed.
%
%   Note: This function relies on MATLAB's in-place semantics for performance,
%         which are not currently supported by Octave.
%
%   See also SPPUSH2.

function [I,J,V,nz] = sppush3(I,J,V,nz,i,j,v)
  N = length(I);
  n = numel(i);
  assert(length(J) == N && length(V) == N && numel(j) == n && numel(v) == n, ...
         'FLAM:sppush3:sizeMismatch', ...
         'Arrays I, J, and V must have the same size.')
  nznew = nz + n;
  if N < nznew
    while N < nznew, N = 2*N; end  % exponentially increase capacity as needed
    e = zeros(N-length(I),1);
    I = [I; e];
    J = [J; e];
    V = [V; e];
  end
  I(nz+(1:n)) = i;
  J(nz+(1:n)) = j;
  V(nz+(1:n)) = v;
  nz = nznew;
end