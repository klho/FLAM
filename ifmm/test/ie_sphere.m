% Second-kind integral equation on the unit sphere, Laplace double-layer.
%
% This is essentially the 3D version of IE_CIRCLE, where now the geometry is the
% unit sphere represented using flat triangles. The system is discretized via
% collocation with tensor-product Gauss-Legendre integration for the near-field
% panel quadratures (within a distance of approximately one triangle size) and
% simple point-to-point interactions for the far-field ones. The resulting
% matrix is square, real, and unsymmetric.

function ie_sphere(n,nquad,occ,p,rank_or_tol,near,store)

  % set default parameters
  if nargin < 1 || isempty(n), n = 20480; end  % number of triangles
  if nargin < 2 || isempty(nquad), nquad = 4; end  % quadrature order
  if nargin < 3 || isempty(occ), occ = 1024; end
  if nargin < 4 || isempty(p), p = 512; end  % number of proxy points
  if nargin < 5 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 6 || isempty(near), near = 0; end  % no near-field compression
  if nargin < 7 || isempty(store), store = 'a'; end  % store all interactions

  % initialize
  [V,F] = trisphere_subdiv(n);  % vertices and faces of triangle discretization
  [x,nu,area] = tri3geom(V,F);  % centroid, normal, and area of each triangle
  N = size(x,2);
  % proxy points are quasi-uniform sampling of scaled 1.5-radius sphere
  proxy = trisphere_subdiv(p);
  % reference proxy points are for unit box [-1, 1]^3

  % compute near-field quadratures
  tic
  if nquad > 0

    % generate reference transformations for each triangle
    [trans,rot,V2,V3] = tri3transrot(V,F);

    % initialize quadrature on the unit square
    [xq,wq] = glegquad(nquad,0,1);
    [xq,yq] = meshgrid(xq); wq = wq*wq';  % tensor product rule
    xq = xq(:); yq = yq(:); wq = wq(:);

    % find neighbors of each triangle
    lrt = 2;                   % root node size
    h = sqrt(8*pi/N);          % average triangle length
    nlvl = ceil(log2(lrt/h));  % number of tree levels to triangle length scale
    T = hypoct(x,0,nlvl);      % build tree to that level

    % initialize sparse matrix storage
    nz = 0;
    for i = T.lvp(nlvl)+1:T.lvp(nlvl+1)
      node = T.nodes(i);
      nslf = length(node.xi);
      nnbr = length([T.nodes(node.nbor).xi]);
      % for simplicity, ignore diagonal values -- will just set to identity
      nz = nz + nslf*(nslf + nnbr - 1);
    end
    I = zeros(nz,1); J = zeros(nz,1); S = zeros(nz,1);

    % compute integrals
    nz = 0;
    for i = T.lvp(nlvl)+1:T.lvp(nlvl+1)
      node = T.nodes(i);
      nbor = [T.nodes(node.nbor).xi];
      for k = node.xi
        % map quadrature rule from square to reference triangle
        [X,Y,W] = qmap_sqtri2(xq,yq,wq,V2(k),V3(:,k));
        for j = [node.xi nbor]
          if j == k, continue; end                 % skip self-interaction
          trg = rot(:,:,k)*(x(:,j) + trans(:,k));  % target in reference space
          q = W'*quadfun(X,Y,trg);                 % apply quadrature
          nz = nz + 1;
          I(nz) = j; J(nz) = k; S(nz) = q;
        end
      end
    end
  else  % skip quadratures -- just use area-weighted point interactions
    I = []; J = []; S = [];
  end
  S = sparse(I,J,S,N,N);  % store in sparse matrix
  t = toc;
  w = whos('S'); mem = w.bytes/1e6;
  fprintf('quad time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)
  clear V F trans rot V2 V3 T I J

  % compress matrix
  Afun = @(i,j)Afun_(i,j,x,nu,area,S);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,nu, ...
                                            area);
  opts = struct('near',near,'store',store,'verb',1);
  tic; F = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('ifmm time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test matrix apply accuracy
  X = rand(N,1); X = X/norm(X);
  tic; ifmm_mv(F,X,Afun,'n'); t = toc;  % for timing
  X = rand(N,16); X = X/norm(X);  % test against 16 vectors for robustness
  r = randperm(N); r = r(1:min(N,128));  % check up to 128 rows in result
  Y = ifmm_mv(F,X,Afun,'n');
  Z = Afun(r,1:N)*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('ifmm_mv:\n')
  fprintf('  multiply err/time: %10.4e / %10.4e (s)\n',err,t)

  % test matrix adjoint apply accuracy
  X = rand(N,1); X = X/norm(X);
  tic; ifmm_mv(F,X,Afun,'c'); t = toc;  % for timing
  X = rand(N,16); X = X/norm(X);  % test against 16 vectors for robustness
  r = randperm(N); r = r(1:min(N,128));  % check up to 128 rows in result
  Y = ifmm_mv(F,X,Afun,'c');
  Z = Afun(1:N,r)'*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('  adjoint multiply err/time: %10.4e / %10.4e (s)\n',err,t)

  % generate field due to exterior sources (PDE reference solution)
  m = 16;                 % number of sources
  src = randn(3,m);                                % random source points on
  src = 2*bsxfun(@rdivide,src,sqrt(sum(src.^2)));  % outer radius-2 sphere
  q = rand(m,1);          % random charges for each source point
  B = Kfun(x,src,'s')*q;  % field evaluated at surface

  % solve for surface density
  tic; [X,~,~,iter] = gmres(@(x)ifmm_mv(F,x,Afun),B,[],1e-6,32); t = toc;
  r = randperm(N); r = r(1:min(N,128));
  err = norm(B(r) - Afun(r,1:N)*X)/norm(B(r));
  fprintf('gmres resid/iter/time: %10.4e / %4d / %10.4e (s)\n',err,iter(2),t)

  % evaluate field from solved density at interior targets
  trg = randn(3,m);                                  % random target points on
  trg = 0.5*bsxfun(@rdivide,trg,sqrt(sum(trg.^2)));  % inner radius-0.5 sphere
  Y = bsxfun(@times,Kfun(trg,x,'d',nu),area)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  err = norm(Z - Y)/norm(Z);
  fprintf('pde solve err: %10.4e\n',err)
end

% quadrature function on reference geometry
function f = quadfun(x,y,trg)
  dx = trg(1) - x;
  dy = trg(2) - y;
  dz = trg(3);
  dr = sqrt(dx.^2 + dy.^2 + dz.^2);
  f = 1/(4*pi).*dz./dr.^3;  % double-layer, normal in z-direction
end

% kernel function
function K = Kfun(x,y,lp,nu)
  dx = bsxfun(@minus,x(1,:)',y(1,:));
  dy = bsxfun(@minus,x(2,:)',y(2,:));
  dz = bsxfun(@minus,x(3,:)',y(3,:));
  dr = sqrt(dx.^2 + dy.^2 + dz.^2);
  if strcmpi(lp,'s')      % single-layer: G
    K = 1/(4*pi)./dr;
  elseif strcmpi(lp,'d')  % double-layer: dG/dn
    rdotn = bsxfun(@times,dx,nu(1,:)) + bsxfun(@times,dy,nu(2,:)) + ...
            bsxfun(@times,dz,nu(3,:));
    K = 1/(4*pi).*rdotn./dr.^3;
  end
end

% matrix entries
function A = Afun_(i,j,x,nu,area,S)
  % quick return if empty
  if isempty(i) || isempty(j), A = zeros(length(i),length(j)); return; end
  % area-weighted point interaction
  A = bsxfun(@times,Kfun(x(:,i),x(:,j),'d',nu(:,j)),area(j));
  % replace near-field with precomputed quadratures
  M = spget(S,i,j); nzidx = M ~= 0; A(nzidx) = M(nzidx);
  % replace diagonal with identity -- note: this is a crude approximation and
  % basically ignores the fine behavior of the (removable) kernel singularity
  [I,J] = ndgrid(i,j); A(I == J) = -0.5;
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,nu,area)
  pxy = bsxfun(@plus,proxy*l,ctr');  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  if strcmpi(rc,'r')
    % from proxy points to centroids: use average triangle area
    N = size(rx,2); Kpxy = Kfun(rx(:,slf),pxy,'s')*(4*pi/N);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
    dz = cx(3,nbr) - ctr(3);
  else
    % from triangles to proxy points: use actual triangle areas
    Kpxy = bsxfun(@times,Kfun(pxy,cx(:,slf),'d',nu(:,slf)),area(slf));
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
    dz = cx(3,nbr) - ctr(3);
  end
  % proxy points form sphere of scaled radius 1.5 around current box
  % keep among neighbors only those within sphere
  dist = sqrt(dx.^2 + dy.^2 + dz.^2);
  nbr = nbr(dist/l < 1.5);
end