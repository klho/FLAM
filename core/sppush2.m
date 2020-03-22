% SPPUSH2  Append to 2-array sparse matrix representation.
%
%   [I,J,NZ] = SPPUSH2(I,J,NZ,IA,JA) overwrites I(NZ+(1:N)) = IA and
%   J(NZ+(1:N)) = JA, where N = LENGTH(IA) = LENGTH(JA), and replaces
%   NZ := NZ + N, appropriately reallocating and expanding the capacities of I
%   and J as needed.
%
%   Note: This function relies on MATLAB's in-place semantics for performance,
%         which are not currently supported by Octave.
%
%   See also SPPUSH3.

function [I,J,nz] = sppush2(I,J,nz,i,j)
  N = length(I);
  n = numel(i);
  assert(length(J) == N && numel(j) == n,'FLAM:sppush2:sizeMismatch', ...
         'Arrays I and J must have the same size.')
  nznew = nz + n;
  if N < nznew
    while N < nznew, N = 2*N; end  % exponentially increase capacity as needed
    e = zeros(N-length(I),1);
    I = [I; e];
    J = [J; e];
  end
  I(nz+(1:n)) = i;
  J(nz+(1:n)) = j;
  nz = nznew;
end