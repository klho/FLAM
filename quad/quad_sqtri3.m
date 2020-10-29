% QUAD_SQTRI3  Convert a quadrature on the unit square to one on a 3D triangle.
%
%    [X,W] = QUAD_SQTRI3(X0,W0,V) produces quadrature nodes and weights X and W,
%    respectively, on the triangle with vertices V by mapping from a quadrature
%    rule on the unit square with nodes and weights X0 and W0, respectively.

function [x,w] = quad_sqtri3(x,w,v)

  % Duffy transform to unit triangle in 2D
  x(2,:) = x(1,:).*x(2,:);
  w = w.*x(1,:)';

  % map unit triangle in 2D to given triangle in 3D
  A = [v(:,2)-v(:,1), v(:,3)-v(:,2)];
  x = A*x + v(:,1);
  [~,A] = qr(A,0);
  w = w*abs(det(A));
end