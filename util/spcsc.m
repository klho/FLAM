% SPCSC Convert from native MATLAB sparse format to explicit CSC format (native
%       MATLAB sparse access appears to be slow for large matrices).
%
%    [AI,AX,P] = SPCSC(A) produces CSC row and data arrays AI and AX,
%    respectively, corresponding to the sparse matrix A, each of which is an
%    N x 1 cell array; and an M x 1 work array P, where [M,N] = SIZE(A).
%
%    See also SPSMAT.

function [Ai,Ax,P] = spcsc(A)

  % initialize
  [m,n] = size(A);
  Ai = cell(n,1);
  Ax = cell(n,1);
  P = zeros(m,1);

  % extract each column
  for i = 1:n
    [Ai{i},~,Ax{i}] = find(A(:,i));
  end
end