% Nine-point stencil on the unit square, constant-coefficient Poisson equation,
% Dirichlet boundary conditions.
%
% This is basically the same as FD_SQUARE1X but using a fourth-order
% approximation with stencil radius two. Correspondingly, the nested dissection
% separators have width two.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-9)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SKIP: skip parameter (default: SKIP = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - DOITER: whether to run unpreconditioned CG (default: DOITER = 1)
%   - DIAGMODE: diagonal extraction mode - 0: skip; 1: matrix unfolding; 2:
%       sparse apply/solves (default: DIAGMODE = 0)

function fd_square4x(n,occ,rank_or_tol,Tmax,skip,symm,doiter,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(rank_or_tol), rank_or_tol = 1e-9; end
  if nargin < 4 || isempty(Tmax), Tmax = 2; end
  if nargin < 5 || isempty(skip), skip = 2; end
  if nargin < 6 || isempty(symm), symm = 'p'; end
  if nargin < 7 || isempty(doiter), doiter = 1; end
  if nargin < 8 || isempty(diagmode), diagmode = 0; end

  % initialize
  [x1,x2] = ndgrid((1:n-1)/n); x = [x1(:) x2(:)]'; clear x1 x2  % grid points
  N = size(x,2);

  % set up sparse matrix
  % index mapping to each point, including "ghost" points
  idx = zeros(n+3,n+3);
  idx(3:n+1,3:n+1) = reshape(1:N,n-1,n-1);
  mid  = 3:n+1;  % "middle"       indices -- interaction with self
  lft1 = 2:n;    % "left  by one" indices -- interaction with one below
  lft2 = 1:n-1;  % "left  by two" indices -- interaction with two below
  rgt1 = 4:n+2;  % "right by one" indices -- interaction with one above
  rgt2 = 5:n+3;  % "right by two" indices -- interaction with two above
  I = idx(mid,mid); e = ones(size(I));
  % interactions with ...
  Jl1 = idx(lft1,mid); Sl1 = -4/3*e;  % ... left-one
  Jl2 = idx(lft2,mid); Sl2 = 1/12*e;  % ... left-two
  Jr1 = idx(rgt1,mid); Sr1 = -4/3*e;  % ... right-one
  Jr2 = idx(rgt2,mid); Sr2 = 1/12*e;  % ... right-two
  Ju1 = idx(mid,lft1); Su1 = -4/3*e;  % ... up-one
  Ju2 = idx(mid,lft2); Su2 = 1/12*e;  % ... up-two
  Jd1 = idx(mid,rgt1); Sd1 = -4/3*e;  % ... down-one
  Jd2 = idx(mid,rgt2); Sd2 = 1/12*e;  % ... down-two
  % ... middle (self)
  Jm = idx(mid,mid); Sm = -(Sl1 + Sl2 + Sr1 + Sr2 + Su1 + Su2 + Sd1 + Sd2);
  % combine all interactions
  I = [  I(:);   I(:);   I(:);   I(:);   I(:);   I(:);   I(:);   I(:);  I(:)];
  J = [Jl1(:); Jl2(:); Jr1(:); Jr2(:); Ju1(:); Ju2(:); Jd1(:); Jd2(:); Jm(:)];
  S = [Sl1(:); Sl2(:); Sr1(:); Sr2(:); Su1(:); Su2(:); Sd1(:); Sd2(:); Sm(:)];
  % remove ghost interactions
  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Jl1 Sl1 Jl2 Sl2 Jr1 Sr1 Jr2 Sr2 Ju1 Su1 Ju2 Su2 Jd1 Sd1 Jd2 Sd2 ...
            Jm Sm I J S

  % factor matrix
  opts = struct('Tmax',Tmax,'skip',skip,'symm',symm,'verb',1);
  tic; F = hifde2x(A,x,occ,rank_or_tol,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifde2x time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; hifde_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(A*x - hifde_mv(F,x)),[],[],1);
  err = err/snorm(N,@(x)(A*x),[],[],1);
  fprintf('hifde_mv: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; hifde_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - A*hifde_sv(F,x)),@(x)(x - hifde_sv(F,A*x,'c')));
  fprintf('hifde_sv: %10.4e / %10.4e (s)\n',err,t)

  % test Cholesky accuracy -- error is w.r.t. compressed apply/solve
  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic; hifde_cholmv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(hifde_mv(F,x) ...
                     - hifde_cholmv(F,hifde_cholmv(F,x,'c'))),[],[],1);
    err = err/snorm(N,@(x)hifde_mv(F,x),[],[],1);
    fprintf('hifde_cholmv: %10.4e / %10.4e (s)\n',err,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic; hifde_cholsv(F,X); t = toc;  % for timing
    err = snorm(N,@(x)(hifde_sv(F,x) ...
                     - hifde_cholsv(F,hifde_cholsv(F,x),'c')),[],[],1);
    err = err/snorm(N,@(x)hifde_sv(F,x),[],[],1);
    fprintf('hifde_cholsv: %10.4e / %10.4e (s)\n',err,t)
  end

  % run unpreconditioned CG
  B = A*X;
  iter = nan;
  if doiter, [~,~,~,iter] = pcg(@(x)(A*x),B,1e-12,128); end

  % run preconditioned CG
  tic; [Y,~,~,piter] = pcg(@(x)(A*x),B,1e-12,32,@(x)hifde_sv(F,x)); t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - A*Y)/norm(B);
  fprintf('cg:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n', ...
          err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',piter,iter)

  % compute log-determinant
  tic; ld = hifde_logdet(F); t = toc;
  fprintf('hifde_logdet:\n')
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
    if diagmode == 1, fprintf('hifde_diag:\n')
    else,             fprintf('hifde_spdiag:\n')
    end

    % extract diagonal
    tic;
    if diagmode == 1, D = hifde_diag(F,0,opts);
    else,             D = hifde_spdiag(F);
    end
    t = toc;
    Y = hifde_mv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  fwd: %10.4e / %10.4e (s)\n',err,t)

    % extract diagonal of inverse
    tic;
    if diagmode == 1, D = hifde_diag(F,1,opts);
    else,             D = hifde_spdiag(F,1);
    end
    t = toc;
    Y = hifde_sv(F,X);
    for i = 1:m, E(i) = Y(r(i),i); end
    err = norm(D(r) - E)/norm(E);
    fprintf('  inv: %10.4e / %10.4e (s)\n',err,t)
  end
end