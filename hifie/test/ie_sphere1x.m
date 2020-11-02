% Second-kind integral equation on the unit sphere, Laplace double-layer.
%
% This is the same as IE_SPHERE1 but using HIFIE3X.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: minimum number of triangles from subdivision (default: N = 20480)
%   - NQUAD: 1D quadrature order (default: NQUAD = 4)
%   - OCC: tree occupancy parameter (default: OCC = 1024)
%   - P: number of proxy points (default: P = 512)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-3)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SKIP: skip parameter (default: SKIP = 0)
%   - STORE: FMM storage mode (default: STORE = 'A')

function ie_sphere1x(n,nquad,occ,p,rank_or_tol,Tmax,skip,store)

  % set default parameters
  if nargin < 1 || isempty(n), n = 20480; end
  if nargin < 2 || isempty(nquad), nquad = 4; end
  if nargin < 3 || isempty(occ), occ = 1024; end
  if nargin < 4 || isempty(p), p = 512; end
  if nargin < 5 || isempty(rank_or_tol), rank_or_tol = 1e-3; end
  if nargin < 6 || isempty(Tmax), Tmax = 2; end
  if nargin < 7 || isempty(skip), skip = 1; end
  if nargin < 8 || isempty(store), store = 'a'; end

  % initialize
  [V,F] = trisphere_subdiv(n);  % vertices and faces of triangle discretization
  [x,nu,area] = tri3geom(V,F);  % centroid, normal, and area of each triangle
  N = size(x,2);
  % proxy points are quasi-uniform sampling of scaled 1.5-radius sphere
  proxy = trisphere_subdiv(p,'v'); r = randperm(size(proxy,2));
  proxy = proxy(:,r(1:p));  % reference proxy points are for unit box [-1, 1]^3

  % compute near-field quadratures
  tic
  if nquad > 0

    % initialize quadrature on the unit square
    [xq,wq] = glegquad(nquad,0,1);
    [xq,yq] = ndgrid(xq); wq = wq*wq';  % tensor product rule
    xq = [xq(:) yq(:)]'; wq = wq(:);

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
        j = setdiff([node.xi nbor],k);             % skip self-interaction
        [Xq,Wq] = quad_sqtri3(xq,wq,V(:,F(:,k)));  % map quadrature to triangle
        K = Kfun(x(:,j),Xq,'d',nu(:,k))*Wq;        % apply quadrature
        n = length(j);
        I(nz+(1:n)) = j; J(nz+(1:n)) = k; S(nz+(1:n)) = K;
        nz = nz + n;
      end
    end
  else  % skip quadratures -- just use area-weighted point interactions
    I = []; J = []; S = [];
  end
  S = sparse(I,J,S,N,N);  % store in sparse matrix
  t = toc;
  w = whos('S'); mem = w.bytes/1e6;
  fprintf('quad time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)
  clear V F T I J

  % factor matrix using HIFIE
  Afun = @(i,j)Afun_(i,j,x,nu,area,S);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy,nu,area);
  opts = struct('Tmax',Tmax,'skip',skip,'verb',1);
  tic; F = hifie3x(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifie3x time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % compress matrix using IFMM
  rank_or_tol = max(rank_or_tol*1e-2,1e-15);  % higher accuracy for reference
  pxyfun_ifmm = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_ifmm_(rc,rx,cx,slf,nbr,l, ...
                                                      ctr,proxy,nu,area);
  opts = struct('Tmax',Tmax,'store',store);
  tic; G = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun_ifmm,opts); t = toc;
  w = whos('G'); mem = w.bytes/1e6;
  fprintf('ifmm time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; hifie_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(ifmm_mv(G,x,Afun,'n') - hifie_mv(F,x,'n')), ...
                @(x)(ifmm_mv(G,x,Afun,'c') - hifie_mv(F,x,'c')));
  err = err/snorm(N,@(x)ifmm_mv(G,x,Afun,'n'),@(x)ifmm_mv(G,x,Afun,'c'));
  fprintf('hifie_mv err/time: %10.4e / %10.4e (s)\n',err,t)
  tic; ifmm_mv(G,X,Afun); t = toc;
  fprintf('ifmm_mv time: %10.4e (s)\n',t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; hifie_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - ifmm_mv(G,hifie_sv(F,x,'n'),Afun,'n')), ...
                @(x)(x - hifie_sv(F,ifmm_mv(G,x,Afun,'c'),'c')));
  fprintf('hifie_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  % generate field due to exterior sources (PDE reference solution)
  m = 16;                 % number of sources
  src = randn(3,m);                % random source points on ...
  src = 2*src./sqrt(sum(src.^2));  % ... outer radius-2 sphere
  q = rand(m,1);          % random charges for each source point
  B = Kfun(x,src,'s')*q;  % field evaluated at surface

  % solve for surface density
  X = hifie_sv(F,B);

  % evaluate field from solved density at interior targets
  trg = randn(3,m);                  % random target points on ...
  trg = 0.5*trg./sqrt(sum(trg.^2));  % ... inner radius-0.5 sphere
  Y = (Kfun(trg,x,'d',nu).*area)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  err = norm(Z - Y)/norm(Z);
  fprintf('pde solve err: %10.4e\n',err)
end

% kernel function
function K = Kfun(x,y,lp,nu)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  dz = x(3,:)' - y(3,:);
  dr = sqrt(dx.^2 + dy.^2 + dz.^2);
  if lp == 's'      % single-layer: G
    K = 1/(4*pi)./dr;
  elseif lp == 'd'  % double-layer: dG/dn
    rdotn = dx.*nu(1,:) + dy.*nu(2,:) + dz.*nu(3,:);
    K = 1/(4*pi)*rdotn./dr.^3;
  end
end

% matrix entries
function A = Afun_(i,j,x,nu,area,S)
  % quick return if empty
  if isempty(i) || isempty(j), A = zeros(length(i),length(j)); return; end
  % area-weighted point interaction
  A = Kfun(x(:,i),x(:,j),'d',nu(:,j)).*area(j);
  % replace near-field with precomputed quadratures
  M = spget(S,i,j); nzidx = M ~= 0; A(nzidx) = M(nzidx);
  % replace diagonal with identity (double-layer vanishes on flat triangles)
  [I,J] = ndgrid(i,j); A(I == J) = -0.5;
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy,nu,area)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  N = size(x,2);  % from proxy points to centroids: use average triangle area
  Kpxy_r = Kfun(x(:,slf),pxy,'s')*(4*pi/N);
  % from triangles to proxy points: use actual triangle areas
  Kpxy_c = Kfun(pxy,x(:,slf),'d',nu(:,slf)).*area(slf);
  Kpxy = [Kpxy_c; Kpxy_r'];  % stack row/column proxy matrices
  % proxy points form ellipsoid of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipsoid
  nbr = nbr(sum(((x(:,nbr) - ctr)./l).^2) < 1.5^2);
end

% proxy function for IFMM
function [Kpxy,nbr] = pxyfun_ifmm_(rc,rx,cx,slf,nbr,l,ctr,proxy,nu,area)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  if rc == 'r'
    % from proxy points to centroids: use average triangle area
    N = size(rx,2);
    Kpxy = Kfun(rx(:,slf),pxy,'s')*(4*pi/N);
    dr = cx(:,nbr) - ctr;
  else
    % from triangles to proxy points: use actual triangle areas
    Kpxy = Kfun(pxy,cx(:,slf),'d',nu(:,slf)).*area(slf);
    dr = rx(:,nbr) - ctr;
  end
  % proxy points form ellipsoid of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipsoid
  nbr = nbr(sum((dr./l).^2) < 1.5^2);
end