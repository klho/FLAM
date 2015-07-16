% GLEGQUAD   Gauss-Legendre quadrature nodes and weights.
%
%    [X,W] = GLEGQUAD(N) produces the nodes and weights X and W, respectively,
%    of the Gauss-Legendre quadrature rule of order N on [-1,1].
%
%    [X,W] = GLEGQUAD(N,A,B) produces the quadrature rule on [A,B].
%
%    See also GPGW.

function [x,w] = glegquad(n,a,b)

  % set default parameters
  if nargin < 2 || isempty(a)
    a = -1;
  end
  if nargin < 3 || isempty(b)
    b = 1;
  end

  % check inputs
  assert(n >= 1,'FLAM:glegquad:invalidOrder', ...
         'Quadrature order must be at least 1.')

  % initialize
  alpha = zeros(n,1);
  beta = 0.5./sqrt(1 - (2*(1:n-1)).^(-2));
  mu = 2;

  % compute quadratures
  [x,w] = gqgw(alpha,beta,mu);

  % rescale nodes and weights
  x = 0.5*((b - a)*x + a + b);
  w = 0.5* (b - a)*w;
end