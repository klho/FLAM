% SPSMAT        Submatrix from explicit CSC sparse matrix.
%
%    A = SPSMAT(AI,AX,P,I,J) produces the full submatrix A = S(I,J), where S is
%    the sparse matrix defined by the CSC row and data arrays AI and AX,
%    respectively, and the work array P.
%
%    See also SPCSC.

function A = spsmat(Ai,Ax,P,I,J)

  % initialize
  m = length(I);
  n = length(J);
  [Isrt,E] = sort(I);
  P(Isrt) = E;
  A = zeros(m,n);

  % fill nonzeros
  for i = 1:n
    I_ = Ai{J(i)};
    idx = ismembc(I_,Isrt);
    A(P(I_(idx)),i) = Ax{J(i)}(idx);
  end
end