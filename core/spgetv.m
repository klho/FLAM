% SPGETV  Sparse matrix access for sparse column cell array.
%
%    S = SPGETV(A,I,J) returns the equivalent of S = FULL(M(I,J)), where M is
%    the sparse matrix corresponding to the data in A, which is an alternative
%    storage format consisting of an array of individual sparse matrices of unit
%    column dimension, one for each column in M, i.e., A{J} = M(:,J).
%
%    See also SPADDV.

function S = spgetv(A,I,J)
  n = numel(J);
  S = zeros(numel(I),n);
  for j = 1:n, S(:,j) = A{J(j)}(I); end
end