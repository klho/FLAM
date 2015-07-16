% TRI3GEOM   Centroid, unit normal, and area of a triangle in 3D.
%
%    [C,N,A] = TRI3GEOM(V) produces the centroid C, unit normal N, and area A
%    of the triangle with vertices V.
%
%    [C,N,A] = TRI3GEOM(V,F) computes the centroid, normal, and area of each
%    triangle I with vertices V(:,F(:,I)).

function [C,N,A] = tri3geom(V,F)

  % set default parameters
  if nargin < 2 || isempty(F)
    F = [1; 2; 3];
  end

  % compute triangle information
  C = (V(:,F(1,:)) + V(:,F(2,:)) + V(:,F(3,:)))/3;
  V21 = V(:,F(2,:)) - V(:,F(1,:));
  V32 = V(:,F(3,:)) - V(:,F(2,:));
  N = cross(V21,V32);
  A = sqrt(sum(N.^2));
  N = bsxfun(@rdivide,N,A);
  A = 0.5*A;
end