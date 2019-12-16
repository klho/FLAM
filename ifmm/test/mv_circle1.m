% Source-target evaluations on the unit circle, Laplace kernel.
%
% This example computes the interactions between random source and target points
% on the unit circle via the Laplace kernel. The associated matrix is
% rectangular and real.
%
% This demo does the following in order:
%
%   - compress the matrix
%   - check multiply error/time
%   - check adjoint multiply error/time

function mv_circle1(M,N,occ,p,rank_or_tol,near,store)

  % set default parameters
  if nargin < 1 || isempty(M), M = 16384; end  % number of row points
  if nargin < 2 || isempty(N), N =  8192; end  % number of col points
  if nargin < 3 || isempty(occ), occ = 128; end
  if nargin < 4 || isempty(p), p = 64; end  % number of proxy points
  if nargin < 5 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 6 || isempty(near), near = 0; end  % no near-field compression
  if nargin < 7 || isempty(store), store = 'n'; end  % no storage

  % initialize
  theta = 2*pi*rand(1,M); rx = [cos(theta); sin(theta)];       % row points
  theta = 2*pi*rand(1,N); cx = [cos(theta); sin(theta)];       % col points
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % compress matrix
  Afun = @(i,j)Afun_(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy);
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
function K = Kfun(x,y)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
end

% matrix entries
function A = Afun_(i,j,rx,cx)
  A = Kfun(rx(:,i),cx(:,j));
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = proxy*l + ctr;  % scale and translate reference points
  if rc == 'r'
    Kpxy = Kfun(rx(:,slf),pxy);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  else
    Kpxy = Kfun(pxy,cx(:,slf));
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  % proxy points form circle of scaled radius 1.5 around current box
  % keep among neighbors only those within circle
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end