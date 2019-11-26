% TRISPHERE_SUBDIV  Recursive subdivision of unit sphere triangulation.
%
%    [V,F] = TRISPHERE_SUBDIV(N) produces a triangulation of the unit sphere
%    with vertices V and faces F of size at least N by recursively subdividing a
%    base icosahedron triangulation. The vertices of triangle I are V(:,F(:,I)).
%    The number of triangles on output can be 20, 80, 320, 1280, etc.
%
%    [V,F] = TRISPHERE_SUBDIV(N,VFMIN) produces a triangulation of size at least
%    N, where size is measured in terms of vertices if VFMIN = 'V' and of faces
%    if VFMIN = 'F'.

function [V,F] = trisphere_subdiv(n,vfmin)

  % set default parameters
  if nargin < 2 || isempty(vfmin), vfmin = 'f'; end

  % check inputs
  vfmin = lower(vfmin);
  assert(strcmp(vfmin,'v') || strcmp(vfmin,'f'), ...
         'FLAM:trisphere_subdiv:invalidVfmin', ...
         'Minimum vertex/face parameter must be one of ''V'' or ''F''.')

  % base triangulation
  t = (1 + sqrt(5))/2;
  V = [-1  t  0
        1  t  0
       -1 -t  0
        1 -t  0
        0 -1  t
        0  1  t
        0 -1 -t
        0  1 -t
        t  0 -1
        t  0  1
       -t  0 -1
       -t  0 1];
  V = V./norm(V(1,:));
  F = [ 1 12  6
        1  6  2
        1  2  8
        1  8 11
        1 11 12
        2  6 10
        6 12  5
       12 11  3
       11  8  7
        8  2  9
        4 10  5
        4  5  3
        4  3  7
        4  7  9
        4  9 10
        5 10  6
        3  5 12
        7  3 11
        9  7  8
       10  9  2];
  if vfmin == 'v', m = size(V,1);
  else,            m = size(F,1);
  end

  % recursively subdivide and project
  while m < n
    M = 0.5*[V(F(:,1),:) + V(F(:,2),:)
             V(F(:,1),:) + V(F(:,3),:)
             V(F(:,2),:) + V(F(:,3),:)];
    M = M./(sqrt(sum(M.^2,2))*ones(1,3));
    [M,~,I] = unique(M,'rows');
    nv = size(V,1);
    nf = size(F,1);
    F12 = nv + I(     1:  nf);
    F13 = nv + I(  nf+1:2*nf);
    F23 = nv + I(2*nf+1:3*nf);
    V = [V; M];
    F = [F(:,1)  F12    F13
          F12   F(:,2)  F23
          F13    F23   F(:,3)
          F12    F23    F13];
    if vfmin == 'v', m = size(V,1);
    else,            m = size(F,1);
    end
  end

  % rotate outputs
  V = V'; F = F';
end