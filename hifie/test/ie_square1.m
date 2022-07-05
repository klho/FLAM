% First-kind integral equation on the unit square, Laplace single-layer.
%
% This example solves the Poisson equation on the unit square by using a single-
% layer volume potential, which yields a first-kind volume integral equation.
% The problem is discretized on a uniform grid, with simple point-to-point
% quadratures for the off-diagonal entries and brute-force numerical integration
% for the diagonal ones. The resulting matrix is square, real, symmetric, and
% Toeplitz.
%
% This demo does the following in order:
%
%   - factor the matrix
%   - check multiply/solve error/time
%   - compare MINRES with/without preconditioning by approximate solve
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - P: number of proxy points (default: P = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-6)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SKIP: skip parameter (default: SKIP = 1)
%   - SYMM: symmetry parameter (default: SYMM = 'H')
%   - DOITER: whether to run unpreconditioned MINRES (default: DOITER = 1)

function ie_square1(n,occ,p,rank_or_tol,Tmax,skip,symm,doiter)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(p), p = 64; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(skip), skip = 1; end
  if nargin < 7 || isempty(symm), symm = 'h'; end
  if nargin < 8 || isempty(doiter), doiter = 1; end

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2;  % grid points
  N = size(x,2);
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % compute diagonal quadratures
  h = 1/n;
  intgrl = 4*dblquad(@(x,y)(-1/(2*pi)*log(sqrt(x.^2 + y.^2))),0,h/2,0,h/2);

  % factor matrix
  Afun = @(i,j)Afun_(i,j,x,intgrl);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy);
  opts = struct('Tmax',Tmax,'skip',skip,'symm',symm,'verb',1);
  tic; F = hifie2(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifie2 time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = reshape(Afun(1:N,1),n,n);
  B = zeros(2*n-1,2*n-1);  % zero-pad
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(n+1:end,:) = flip(B(n+1:end,:),1);
  B(:,n+1:end) = flip(B(:,n+1:end),2);
  G = fft2(B);
  mv = @(x)mv_(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; hifie_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - hifie_mv(F,x)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('hifie_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; hifie_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - mv(hifie_sv(F,x))),@(x)(x - hifie_sv(F,mv(x),'c')));
  fprintf('hifie_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  B = mv(X);
  do_minres = 1;
  if isoctave()
    warning('No MINRES in Octave; using GMRES.')
    do_minres = 0;
  end

  % run unpreconditioned MINRES
  if do_minres
    iter = nan;
    if doiter, [~,~,~,iter] = minres(mv,B,1e-12,128); end
  else
    iter(1:2) = nan;
    if doiter, [~,~,~,iter] = gmres(mv,B,32,1e-12,32); end
    iter = (iter(1) + 1)*iter(2);  % total iterations
  end

  % run preconditioned MINRES
  tic
  if do_minres
    [Y,~,~,piter] = minres(mv,B,1e-12,32,@(x)hifie_sv(F,x));
  else
    [Y,~,~,piter] = gmres(mv,B,32,1e-12,32,@(x)hifie_sv(F,x));
    piter = (piter(1) + 1)*piter(2);  % total iterations
  end
  t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - mv(Y))/norm(B);
  if do_minres, fprintf('minres:\n')
  else,         fprintf('gmres:\n')
  end
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',piter,iter)
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
  A(I == J) = intgrl;         % replace diagonal with precomputed values
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  N = size(x,2);
  Kpxy = Kfun(pxy,x(:,slf))/N;
  % proxy points form ellipse of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipse
  nbr = nbr(sum(((x(:,nbr) - ctr)./l).^2) < 1.5^2);
end

% FFT multiplication
function y = mv_(F,x)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(x,n,n),2*n-1,2*n-1));
  y = reshape(y(1:n,1:n),N,1);
end