% Covariance matrix on the unit square, Matern 3/2 kernel.
%
% This is basically the same as COV_SQUARE1 but using the Matern 3/2 kernel.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - P: square root of number of proxy points (default: P = 16)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-6)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - NOISE: nugget effect (default: NOISE = 1e-2)
%   - SCALE: kernel length scale (default: SCALE = 100)
%   - DIAGMODE: diagonal extraction mode - 0: skip; 1: matrix unfolding; 2:
%       sparse apply/solves (default: DIAGMODE = 1)

function cov_square2(n,occ,p,rank_or_tol,Tmax,symm,noise,scale,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(p), p = 16; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(symm), symm = 'p'; end
  if nargin < 7 || isempty(noise), noise = 1e-2; end
  if nargin < 8 || isempty(scale), scale = 100; end
  if nargin < 9 || isempty(diagmode), diagmode = 1; end

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2;  % grid points
  N = size(x,2);
  % proxy points -- rectangular annulus
  theta = (1:p)*2*pi/p; proxy_ = [cos(theta); sin(theta)];
  proxy_ = proxy_./max(abs(proxy_));                           % base "ring"
  R = 3/scale;                                                 % annular width
  proxy = reshape(proxy_(:)*linspace(0,R,p),2,p,p);            % proxy points
  % reference proxy points are for a single point at the origin only
  shift = 1.5*proxy_;  % reference shift for the unit box [-1, 1]^2

  % factor matrix
  Afun = @(i,j)Afun_(i,j,x,noise,scale);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy,shift,scale);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskelf time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

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
  tic; rskelf_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - rskelf_mv(F,x)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('rskelf_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; rskelf_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - mv(rskelf_sv(F,x))),@(x)(x - rskelf_sv(F,mv(x),'c')));
  fprintf('rskelf_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  % test Cholesky accuracy -- error is w.r.t. compressed apply/solve
  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic; rskelf_cholmv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(rskelf_mv(F,x) ...
                         - rskelf_cholmv(F,rskelf_cholmv(F,x,'c'))),[],[],1);
    err = err/snorm(N,@(x)rskelf_mv(F,x),[],[],1);
    fprintf('rskelf_cholmv: %10.4e / %10.4e (s)\n',err,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic; rskelf_cholsv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(rskelf_sv(F,x) ...
                         - rskelf_cholsv(F,rskelf_cholsv(F,x),'c')),[],[],1);
    err = err/snorm(N,@(x)rskelf_sv(F,x),[],[],1);
    fprintf('rskelf_cholsv: %10.4e / %10.4e (s)\n',err,t)
  end

  % compute log-determinant
  tic; ld = rskelf_logdet(F); t = toc;
  fprintf('rskelf_logdet:\n')
  fprintf('  real/imag: %22.16e / %22.16e\n',real(ld),imag(ld))
  fprintf('  time: %10.4e (s)\n',t)

  if diagmode > 0
    % prepare for diagonal extraction
    opts = struct('verb',1);
    m = min(N,128);  % number of entries to check against
    r = randperm(N); r = r(1:m);
    % reference comparison from compressed solve against coordinate vectors
    X = zeros(N,m);
    for i = 1:m, X(r(i),i) = 1; end
    E = zeros(m,1);  % solution storage
    if diagmode == 1, fprintf('rskelf_diag:\n')
    else,             fprintf('rskelf_spdiag:\n')
    end

    % extract diagonal
    tic;
    if diagmode == 1, D = rskelf_diag(F,0,opts);
    else,             D = rskelf_spdiag(F);
    end
    t = toc;
    Y = rskelf_mv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  fwd: %10.4e / %10.4e (s)\n',err,t)

    % extract diagonal of inverse
    tic;
    if diagmode == 1, D = rskelf_diag(F,1,opts);
    else,             D = rskelf_spdiag(F,1);
    end
    t = toc;
    Y = rskelf_sv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  inv: %10.4e / %10.4e (s)\n',err,t)
  end
end

% kernel function
function K = Kfun(x,y,scale)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  dr = scale*sqrt(dx.^2 + dy.^2);  % scaled distance
  K = (1 + sqrt(3)*dr).*exp(-sqrt(3)*dr);
end

% matrix entries
function A = Afun_(i,j,x,noise,scale)
  A = Kfun(x(:,i),x(:,j),scale);
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = A(idx) + noise^2;  % modify diagonal with "nugget"
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy,shift,scale)
  pxy = proxy + shift.*l + ctr;  % scale and translate reference points
  Kpxy = Kfun(pxy,x(:,slf),scale);
  % proxy points form "annulus" of scaled inner "radius" 1.5 around current box
  % keep among neighbors only those within annulus
  nbr = nbr(max(abs(x(:,nbr) - ctr)./l) < 1.5);
end

% FFT multiplication
function y = mv_(F,x)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(x,n,n),2*n-1,2*n-1));
  y = reshape(y(1:n,1:n),N,1);
end