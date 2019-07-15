% Source-target evaluations on the unit sphere, Helmholtz kernel.
%
% This is basically the same as MV_SPHERE1 but using the Helmholtz kernel. The
% associated matrix is rectangular and complex.

function mv_sphere2(m,n,k,occ,p,rank_or_tol,near,store)

  % set default parameters
  if nargin < 1 || isempty(m), m = 16384; end  % number of row points
  if nargin < 2 || isempty(n), n =  8192; end  % number of col points
  if nargin < 3 || isempty(k), k = 2*pi*4; end  % wavenumber
  if nargin < 4 || isempty(occ), occ = 256; end
  if nargin < 5 || isempty(p), p = 512; end  % number of proxy points
  if nargin < 6 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 7 || isempty(near), near = 0; end  % no near-field compression
  if nargin < 8 || isempty(store), store = 'n'; end  % no storage

  % initialize
  rx = randn(3,m); rx = bsxfun(@rdivide,rx,sqrt(sum(rx.^2)));  % row points
  cx = randn(3,n); cx = bsxfun(@rdivide,cx,sqrt(sum(cx.^2)));  % col points
  M = size(rx,2);
  N = size(cx,2);
  % proxy points are quasi-uniform sampling of scaled 1.5-radius sphere
  proxy = trisphere_subdiv(p);
  % reference proxy points are for unit box [-1, 1]^3

  % compress matrix
  Afun = @(i,j)Afun_(i,j,rx,cx,k);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,k);
  opts = struct('near',near,'store',store,'verb',1);
  tic; F = ifmm(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('ifmm time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test matrix apply accuracy
  X = rand(N,1); X = X/norm(X);
  tic; ifmm_mv(F,X,Afun,'n'); t = toc;  % for timing
  X = rand(N,16); X = X/norm(X);  % test against 16 vectors for robustness
  r = randperm(M); r = r(1:min(M,128));  % check up to 128 rows in result
  Y = ifmm_mv(F,X,Afun,'n');
  Z = Afun(r,1:N)*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('ifmm_mv:\n')
  fprintf('  multiply err/time: %10.4e / %10.4e (s)\n',err,t)

  % test matrix adjoint apply accuracy
  X = rand(M,1); X = X/norm(X);
  tic; ifmm_mv(F,X,Afun,'c'); t = toc;  % for timing
  X = rand(M,16); X = X/norm(X);  % test against 16 vectors for robustness
  r = randperm(N); r = r(1:min(N,128));  % check up to 128 rows in result
  Y = ifmm_mv(F,X,Afun,'c');
  Z = Afun(1:M,r)'*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('  adjoint multiply err/time: %10.4e / %10.4e (s)\n',err,t)
end

% kernel function
function K = Kfun(x,y,k)
  dx = bsxfun(@minus,x(1,:)',y(1,:));
  dy = bsxfun(@minus,x(2,:)',y(2,:));
  dz = bsxfun(@minus,x(3,:)',y(3,:));
  dr = sqrt(dx.^2 + dy.^2 + dz.^2);
  K = 1/(4*pi).*exp(1i*k*dr)./dr;
end

% matrix entries
function A = Afun_(i,j,rx,cx,k)
  A = Kfun(rx(:,i),cx(:,j),k);
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,k)
  pxy = bsxfun(@plus,proxy*l,ctr');  % scale and translate reference points
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy,k);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
    dz = cx(3,nbr) - ctr(3);
  else
    Kpxy = Kfun(pxy,cx(:,slf),k);
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
    dz = rx(3,nbr) - ctr(3);
  end
  % proxy points form sphere of scaled radius 1.5 around current box
  % keep among neighbors only those within sphere
  dist = sqrt(dx.^2 + dy.^2 + dz.^2);
  nbr = nbr(dist/l < 1.5);
end