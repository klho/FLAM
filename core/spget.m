% SPGET  Sparse matrix access (native MATLAB is slow for large matrices).
%
%    [S,P] = SPGET(A,I,J,P) returns the submatrix S = A(I,J) in dense form using
%    the workspace array P, which must be of size MAX(I). Due to MATLAB
%    semantics, P should always be returned in order to avoid copying it on
%    input.

function [S,P] = spget(A,I,J,P)
  m = length(I);
  n = length(J);
  [I_sort,E] = sort(I);
  P(I_sort) = E;
  S = zeros(m,n);
  [I_,J_,S_] = find(A(:,J));
  idx = ismemb(I_,I_sort);
  I_ = I_(idx);
  J_ = J_(idx);
  S_ = S_(idx);
  S(P(I_) + (J_ - 1)*m) = S_;
end