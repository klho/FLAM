% QMAP_SQTRI2  Convert a quadrature on the unit square to one on a triangle in
%              reference configuration.
%
%    [X,Y,W] = QMAP_SQTRI2(X0,Y0,W0,V2,V3) produces quadrature nodes and weights
%    [X,Y] and W, respectively, on the triangle with vertices [0; 0], [V2; 0],
%    and V3 by mapping from a quadrature on the unit square with nodes and
%    weights [X0,Y0] and W0, respectively.

function [x,y,w] = qmap_sqtri2(x,y,w,v2,v3)

  % Duffy transform to unit triangle
  y = x.*y;
  w = w.*x;

  % map unit triangle to given triangle
  A = zeros(2);
  A(1,1) = v2;
  A(:,2) = v3 - A(:,1);
  z = A*[x y]';
  x = z(1,:)';
  y = z(2,:)';
  w = w*det(A);
end