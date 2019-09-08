% SPADDV  Sparse matrix addition for sparse column cell array.
%
%   A = SPADDV(A,I,J,V) returns the equivalent of M(I,J) := M(I,J) + V, where M
%   is the sparse matrix corresponding to the data stored in A; see SPGETV.
%
%   See also SPGETV.

function A = spaddv(A,I,J,V)
  n = numel(J);
  for j = 1:n, A{J(j)}(I) = A{J(j)}(I) + V(:,j); end
end