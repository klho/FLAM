% Five-point stencil on the unit square, constant-coefficient Poisson equation,
% Dirichlet boundary conditions.
%
% This is the same as FD_SQUARE1 but using MFX.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of points + 1 in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - DOITER: whether to run unpreconditioned CG (default: DOITER = 1)
%   - DIAGMODE: diagonal extraction mode - 0: skip; 1: matrix unfolding; 2:
%       sparse apply/solves (default: DIAGMODE = 0)

function fd_square1x(n,occ,symm,doiter,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(symm), symm = 'p'; end
  if nargin < 4 || isempty(doiter), doiter = 1; end
  if nargin < 5 || isempty(diagmode), diagmode = 0; end

  % initialize
  [x1,x2] = ndgrid((1:n-1)/n); x = [x1(:) x2(:)]'; clear x1 x2  % grid points
  N = size(x,2);

  % set up sparse matrix
  idx = zeros(n+1,n+1);  % index mapping to each point, including "ghost" points
  idx(2:n,2:n) = reshape(1:N,n-1,n-1);
  mid = 2:n;    % "middle" indices -- interaction with self
  lft = 1:n-1;  % "left"   indices -- interaction with one below
  rgt = 3:n+1;  % "right"  indices -- interaction with one above
  I = idx(mid,mid); e = ones(size(I));
  % interactions with ...
  Jl = idx(lft,mid); Sl = -e;                    % ... left
  Jr = idx(rgt,mid); Sr = -e;                    % ... right
  Ju = idx(mid,lft); Su = -e;                    % ... up
  Jd = idx(mid,rgt); Sd = -e;                    % ... down
  Jm = idx(mid,mid); Sm = -(Sl + Sr + Su + Sd);  % ... middle (self)
  % combine all interactions
  I = [ I(:);  I(:);  I(:);  I(:);  I(:)];
  J = [Jl(:); Jr(:); Ju(:); Jd(:); Jm(:)];
  S = [Sl(:); Sr(:); Su(:); Sd(:); Sm(:)];
  % remove ghost interactions
  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Jl Sl Jr Sr Ju Su Jd Sd Jm Sm I J S

  % factor matrix
  opts = struct('symm',symm,'verb',1);
  tic; F = mfx(A,x,occ,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('mfx time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; mf_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(A*x - mf_mv(F,x)),[],[],1);
  err = err/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mf_mv: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; mf_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - A*mf_sv(F,x)),@(x)(x - mf_sv(F,A*x,'c')));
  fprintf('mf_sv: %10.4e / %10.4e (s)\n',err,t)

  % test Cholesky accuracy -- error is w.r.t. compressed apply/solve
  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic; mf_cholmv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(mf_mv(F,x) - mf_cholmv(F,mf_cholmv(F,x,'c'))),[],[],1);
    err = err/snorm(N,@(x)mf_mv(F,x),[],[],1);
    fprintf('mf_cholmv: %10.4e / %10.4e (s)\n',err,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic; mf_cholsv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(mf_sv(F,x) - mf_cholsv(F,mf_cholsv(F,x),'c')),[],[],1);
    err = err/snorm(N,@(x)mf_sv(F,x),[],[],1);
    fprintf('mf_cholsv: %10.4e / %10.4e (s)\n',err,t)
  end

  % run unpreconditioned CG
  B = A*X;
  iter = nan;
  if doiter, [~,~,~,iter] = pcg(@(x)(A*x),B,1e-12,128); end

  % run preconditioned CG
  tic; [Y,~,~,piter] = pcg(@(x)(A*x),B,1e-12,32,@(x)mf_sv(F,x)); t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - A*Y)/norm(B);
  fprintf('cg:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n', ...
          err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',piter,iter)

  % compute log-determinant
  tic; ld = mf_logdet(F); t = toc;
  fprintf('mf_logdet:\n')
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
    if diagmode == 1, fprintf('mf_diag:\n')
    else,             fprintf('mf_spdiag:\n')
    end

    % extract diagonal
    tic;
    if diagmode == 1, D = mf_diag(F,0,opts);
    else,             D = mf_spdiag(F);
    end
    t = toc;
    Y = mf_mv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  fwd: %10.4e / %10.4e (s)\n',err,t)

    % extract diagonal of inverse
    tic;
    if diagmode == 1, D = mf_diag(F,1,opts);
    else,             D = mf_spdiag(F,1);
    end
    t = toc;
    Y = mf_sv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  inv: %10.4e / %10.4e (s)\n',err,t)
  end
end