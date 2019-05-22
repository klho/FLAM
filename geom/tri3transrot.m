% TRI3TRANSROT  Translate and rotate a triangle in 3D to a reference
%               configuration in 2D.
%
%    [TRANS,ROT,V2,V3] = TRI3TRANSROT(V) produces a translation vector TRANS
%    and a rotation matrix ROT such that the affine transformation
%    ROT*(V + TRANS) moves the vertex V(1) to the origin, V(2) to a point
%    [V2; 0] with V2 > 0, and V(3) to a point V3 in the positive quadrant (if
%    positively oriented).
%
%    [TRANS,ROT,V2,V3] = TRI3TRANSROT(V,F) computes this transformation for
%    each triangle I with vertices V(:,F(:,I)).
%
%    See also TRI2TRANSROT.

function [trans,rot,V2,V3] = tri3transrot(V,F)

  % set default parameters
  if nargin < 2 || isempty(F), F = [1; 2; 3]; end

  % recenter at V(:,F(1,:))
  trans = -V(:,F(1,:));
  V21 = V(:,F(2,:)) + trans;
  V31 = V(:,F(3,:)) + trans;

  % store new V(:,F(2,:)) = [V2; 0]
  V2 = sqrt(sum(V21.^2));

  % generate rotation matrices
  n = size(F,2);
  rot = zeros(3,3,n);
  rot(:,1,:) = bsxfun(@rdivide,V21,V2);
  rot(:,3,:) = cross(V21,V31);
  rot(:,3,:) = bsxfun(@rdivide,rot(:,3,:),sqrt(sum(rot(:,3,:).^2)));
  rot(:,2,:) = cross(rot(:,3,:),rot(:,1,:));
  rot = permute(rot,[2 1 3]);

  % compute new V(:,F(3,:))
  V3 = zeros(2,n);
  for i = 1:n, V3(:,i) = rot(1:2,:,i)*V31(:,i); end
end