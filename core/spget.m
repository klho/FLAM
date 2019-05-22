% SPGET  Sparse matrix access.
%
%    S = SPGET(A,I,J) returns S = FULL(A(I,J)).
%
%    This function exists because computing A(I,J) appears to be quite slow in
%    some versions of MATLAB, requiring O(N) time, where N = SIZE(A,2), say (we
%    have only tested square matrices), even when A has only a constant number
%    of nonzeros per row/column and the sizes of I and J are bounded. For such
%    matrices, this implementation instead has cost O(LENGTH(I)*LENGTH(J)).
%
%    This function may not be needed in newer versions of MATLAB. We have not
%    done comprehensive testing.

function S = spget(A,I,J)
  S = A(:,J);
  S = full(S(I,:));
end