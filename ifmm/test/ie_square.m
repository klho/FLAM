% Second-kind integral equation on the unit square, Laplace single-layer.
%
% This example mimics solving the Poisson equation on the unit square by using a
% single-layer volume potential, which would yield a first-kind volume integral
% equation, but we make it second-kind by artificially adding the identity. The
% problem is discretized on a uniform grid, with simple point-to-point
% quadratures for the off-diagonal entries and brute-force numerical integration
% for the diagonal ones. The resulting matrix is square, real, symmetric, and
% Toeplitz.
%
% This demo does the following in order:
%
%   - compress the matrix
%   - check multiply error/time
%   - check solve error/time using GMRES

function ie_square(n,occ,p,rank_or_tol,near,store,symm)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end  % number of points in each dim
  if nargin < 2 || isempty(occ), occ = 128; end
  if nargin < 3 || isempty(p), p = 64; end  % number of proxy points
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-9; end
  if nargin < 5 || isempty(near), near = 0; end  % no near-field compression
  if nargin < 6 || isempty(store), store = 'a'; end  % store all interactions
  if nargin < 7 || isempty(symm), symm = 's'; end  % symmetric

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2;  % grid points
  N = size(x,2);
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % compute diagonal quadratures
  h = 1/n;
  intgrl = 4*dblquad(@(x,y)(-1/(2*pi)*log(sqrt(x.^2 + y.^2))),0,h/2,0,h/2);

  % compress matrix
  Afun = @(i,j)Afun_(i,j,x,intgrl);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'symm',symm,'verb',1);
  tic; F = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('ifmm time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = reshape(Afun(1:N,1),n,n);
  B = zeros(2*n-1,2*n-1);  % zero-pad
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(:,n+1:end) = flipdim(B(:,n+1:end),2);
  B(n+1:end,:) = flipdim(B(n+1:end,:),1);
  G = fft2(B);
  mv = @(x)mv_(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; ifmm_mv(F,X,Afun); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - ifmm_mv(F,x,Afun)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('ifmm_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % run GMRES
  B = mv(X);
  tic; [Y,~,~,iter] = gmres(@(x)ifmm_mv(F,x,Afun),B,[],1e-12,32); t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - mv(Y))/norm(B);
  fprintf('gmres:\n')
  fprintf('  soln/resid err: %10.4e / %10.4e\n',err1,err2)
  fprintf('  iter/time: %d / %10.4e (s)\n',iter(2),t)
end

% kernel function
function K = Kfun(x,y)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
end

% matrix entries
function A = Afun_(i,j,x,intgrl)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j))/N;  % area-weighted point interaction
  [I,J] = ndgrid(i,j);
  A(I == J) = 1 + intgrl;  % replace diagonal with identity + precomputed values
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = proxy*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  N = size(rx,2);
  if rc == 'r'
    Kpxy = Kfun(rx(:,slf),pxy)/N;
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  else
    Kpxy = Kfun(pxy,cx(:,slf))/N;
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  % proxy points form circle of scaled radius 1.5 around current box
  % keep among neighbors only those within circle
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv_(F,x)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(x,n,n),2*n-1,2*n-1));
  y = reshape(y(1:n,1:n),N,1);
end