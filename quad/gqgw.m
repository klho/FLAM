% GQGW   Gaussian quadratures using the Golub-Welsch algorithm.
%
%    [X,W] = GQGW(ALPHA,BETA,MU) produces the nodes and weights X and W,
%    respectively, of the Gaussian quadrature rule corresponding to the
%    orthogonal polynomials P with ALPHA(I) = -B(I)/A(I) and
%    BETA(I) = SQRT(C(I+1)/(A(I)*A(I+1))), where A, B, and C are such that P
%    satisfies the three-term recurrence
%
%      P(I,X) = (A(I)*X + B(I))*P(I-1,X) - C(I)*P(I-2,X)
%
%    for I = 1, 2, 3, ... with P(-1,X) = 0 and P(0,X) = X; and MU is the zeroth
%    moment of the weight function with respect to which P is orthogonal.
%
%    References:
%
%      G.H. Golub, J.H. Welsch. Calculation of Gauss quadrature rules. Math.
%        Comp. 23 (106): 221-230, 1969.

function [x,w] = gqgw(alpha,beta,mu)

  % construct recurrence matrix
  T = diag(beta,-1) + diag(alpha) + diag(beta,1);

  % compute quadrature nodes
  [V,D] = eig(T);
  x = diag(D); [x,i] = sort(x);

  % compute quadrature weights
  w = mu*V(1,i)'.^2;
end