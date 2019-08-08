% Covariance matrix on the unit line, squared exponential kernel.
%
% This example considers the covariance matrix between equispaced points on the
% unit line with squared exponential covariance function. The matrix is square,
% real, positive definite, and Toeplitz.
%
% We remark on a few key departures from the PDE setting:
%
%   - the covariance kernel has a length scale parameter
%   - the covariance matrix has a "nugget" (identity perturbation) to improve
%       conditioning
%   - the proxy points sample a local corona since there is no Green's theorem
%       (but the kernel is real-analytic)
%   - computing the determinant and explicit entries of the inverse covariance
%       matrix (i.e., precision matrix) is of interest
%
% This demo does the following in order:
%
%   - factor the matrix
%   - check multiply/solve error/time
%   - check Cholesky multiply/solve error/time
%   - compute log-determinant
%   - do diagonal inversion (i.e., selected inversion for the diagonal)

function cov_line1(n,occ,p,rank_or_tol,symm,noise,scale,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 16384; end  % number of points
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(p), p = 8; end  % half number of proxy points
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 5 || isempty(symm), symm = 'p'; end  % positive definite
  if nargin < 6 || isempty(noise), noise = 1e-2; end  % nugget effect
  if nargin < 7 || isempty(scale), scale = 100; end  % kernel length scale
  if nargin < 8 || isempty(diagmode), diagmode = 1; end  % diag extraction mode:
  % 0 - skip; 1 - matrix unfolding; 2 - sparse apply/solves

  % initialize
  x = (1:n)/n; N = size(x,2);                           % grid points
  proxy = linspace(1.5,2.5,p); proxy = [-proxy proxy];  % proxy points
  % reference proxy points are for unit box [-1, 1]

  % factor matrix
  Afun = @(i,j)Afun_(i,j,x,noise,scale);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy,scale);
  opts = struct('symm',symm,'verb',1);
  tic; F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskelf time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = Afun(1:n,1);
  B = zeros(2*n-1,1);  % zero-pad
  B(1:n) = a;
  B(n+1:end) = flipud(a(2:n));
  G = fft(B);
  mv = @(x)mv_(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskelf_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - rskelf_mv(F,x)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('rskel_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; rskelf_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - mv(rskelf_sv(F,x))),@(x)(x - rskelf_sv(F,mv(x),'c')));
  fprintf('rskel_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  % test Cholesky accuracy -- error is w.r.t. compressed apply/solve
  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic; rskelf_cholmv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(rskelf_mv(F,x) ...
                         - rskelf_cholmv(F,rskelf_cholmv(F,x,'c'))),[],[],1);
    err = err/snorm(N,@(x)(rskelf_mv(F,x)),[],[],1);
    fprintf('rskelf_cholmv: %10.4e / %10.4e (s)\n',err,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic; rskelf_cholsv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(rskelf_sv(F,x) ...
                         - rskelf_cholsv(F,rskelf_cholsv(F,x),'c')),[],[],1);
    err = err/snorm(N,@(x)(rskelf_sv(F,x)),[],[],1);
    fprintf('rskelf_cholsv: %10.4e / %10.4e (s)\n',err,t)
  end

  % compute log-determinant
  tic; ld = rskelf_logdet(F); t = toc;
  fprintf('rskelf_logdet: %22.16e / %10.4e (s)\n',ld,t)

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
  dr = scale*abs(bsxfun(@minus,x',y));  % scaled distance
  K = exp(-0.5*dr.^2);
end

% matrix entries
function A = Afun_(i,j,x,noise,scale)
  A = Kfun(x(:,i),x(:,j),scale);
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = A(idx) + noise^2;  % modify diagonal with "nugget"
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy,scale)
  pxy = bsxfun(@plus,proxy*l,ctr');  % scale and translate reference points
  Kpxy = Kfun(pxy,x(slf),scale);
  % proxy points form interval of scaled radius 1.5 around current box
  % keep among neighbors only those within interval
  dist = abs(x(nbr) - ctr);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv_(F,x)
  n = length(x);
  y = ifft(F.*fft(x,2*n-1));
  y = y(1:n);
end