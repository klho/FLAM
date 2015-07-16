% ID   Interpolative decomposition.
%
%    [SK,RD,T] = ID(A,K) produces a rank-K approximation of A via the skeleton
%    and redundant indices SK and RD, respectively, and an interpolation matrix
%    T such that A(:,RD) = A(:,SK)*T + E, where LENGTH(SK) = K and NORM(E) is of
%    estimated order S(K) for S = SVD(A). If K > N = SIZE(A,2), then K is set
%    equal to N.
%
%    [SK,RD,T] = ID(A,TOL) produces a rank-adaptive approximation such that
%    A(:,RD) = A(:,SK)*T + E, where NORM(E) is of estimated order TOL*NORM(A).
%
%    [SK,RD,T] = ID(A,RANK_OR_TOL) sets K = RANK_OR_TOL if RANK_OR_TOL >= 1 and
%    TOL = RANK_OR_TOL if RANK_OR_TOL < 1.
%
%    [SK,RD,T] = ID(A,RANK_OR_TOL,SRAND) uses random sampling if SRAND = 1
%    (default: SRAND = 1).
%
%    References:
%
%      H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. On the compression of
%        low rank matrices. SIAM J. Sci. Comput. 26(4): 1389-1404, 2005.
%
%    See also QR.

function [sk,rd,T] = id(A,rank_or_tol,srand)

  % set default parameters
  if nargin < 3 || isempty(srand)
    srand = 1;
  end

  % check inputs
  assert(rank_or_tol >= 0,'FLAM:id:negativeRankOrTol', ...
         'Rank or tolerance must be nonnegative.')

  % initialize
  [m,n] = size(A);

  % return if matrix is empty
  if isempty(A)
    sk = [];
    rd = 1:n;
    T = zeros(0,n);
    return
  end

  % sample against Gaussian matrix if too rectangular
  if srand && m > 2*n
    A = randn(n+16,m)*A;
  end

  % compute ID
  [~,R,E] = qr(A,0);
  if rank_or_tol < 1
    k = sum(abs(diag(R)) > abs(R(1))*rank_or_tol);
  else
    k = min(rank_or_tol,n);
  end
  sk = E(1:k);
  rd = E(k+1:end);
  T = R(1:k,1:k)\R(1:k,k+1:end);
end