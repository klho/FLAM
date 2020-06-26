% Five-point stencil on the unit square, constant-coefficient Poisson equation,
% Dirichlet-Neumann boundary conditions.
%
% This is basically the same as FD_SQUARE1 but with Dirichlet boundary
% conditions along the top and bottom sides of the domain, and Neumann boundary
% conditions along the left and right sides. The Neumann conditions are imposed
% as follows:
%
%   - At interior points, we have the standard five-point stencil relation
%     between U(I,J) and its neighbors U(I-1,J), U(I+1,J), U(I,J-1), and
%     U(I,J+1).
%
%   - Near the left boundary, say, U(1,J) depends on the ghost point U(0,J),
%     which is unknown. However, we can use the Neumann condition in conjunction
%     with a second-order one-sided difference to express it in terms of U(1,J)
%     and U(2,J).
%
%   - Repeat similarly with the right boundary, being careful to rescale the
%     boundary rows of the matrix in order to maintain symmetry.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 8)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-9)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SKIP: skip parameter (default: SKIP = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - DOITER: whether to run unpreconditioned CG (default: DOITER = 1)
%   - DIAGMODE: diagonal extraction mode - 0: skip; 1: matrix unfolding; 2:
%       sparse apply/solves (default: DIAGMODE = 0)

function fd_square5(n,occ,rank_or_tol,Tmax,skip,symm,doiter,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 8; end
  if nargin < 3 || isempty(rank_or_tol), rank_or_tol = 1e-9; end
  if nargin < 4 || isempty(Tmax), Tmax = 2; end
  if nargin < 5 || isempty(skip), skip = 2; end
  if nargin < 6 || isempty(symm), symm = 'p'; end
  if nargin < 7 || isempty(doiter), doiter = 1; end
  if nargin < 8 || isempty(diagmode), diagmode = 0; end

  % initialize
  N = (n - 1)^2;  % total number of grid points

  % set up sparse matrix
  idx = zeros(n+1,n+1);  % index mapping to each point, including "ghost" points
  idx(2:n,2:n) = reshape(1:N,n-1,n-1);
  mid = 2:n;    % "middle" indices -- interaction with self
  lft = 1:n-1;  % "left"   indices -- interaction with one below
  rgt = 3:n+1;  % "right"  indices -- interaction with one above
  % interior interactions with ...
  I = idx(mid,mid); e = ones(size(I));
  Jl = idx(lft,mid); Sl = -e;                    % ... left
  Jr = idx(rgt,mid); Sr = -e;                    % ... right
  Ju = idx(mid,lft); Su = -e;                    % ... up
  Jd = idx(mid,rgt); Sd = -e;                    % ... down
  Jm = idx(mid,mid); Sm = -(Sl + Sr + Su + Sd);  % ... middle (self)
  % boundary interactions with ...
  Il = idx(2,mid); Ir = idx(n,mid); e = ones(size(mid));
  Jl1 = idx(2  ,mid); Sl1 = -4/3*e;  % ... left  side, one over
  Jl2 = idx(3  ,mid); Sl2 =  1/3*e;  % ... left  side, two over
  Jr1 = idx(n  ,mid); Sr1 = -4/3*e;  % ... right side, one over
  Jr2 = idx(n-1,mid); Sr2 =  1/3*e;  % ... right side, two over
  % combine all interactions
  I = [  I(:);   I(:);   I(:);   I(:);   I(:);  Il(:);  Il(:);  Ir(:);  Ir(:)];
  J = [ Jl(:);  Jr(:);  Ju(:);  Jd(:);  Jm(:); Jl1(:); Jl2(:); Jr1(:); Jr2(:)];
  S = [ Sl(:);  Sr(:);  Su(:);  Sd(:);  Sm(:); Sl1(:); Sl2(:); Sr1(:); Sr2(:)];
  % remove ghost interactions
  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);
  A = sparse(I,J,S,N,N);
  A([Il Ir],:) = A([Il Ir],:) * 3/2;  % rescale boundary rows to symmetrize
  clear idx Jl Sl Jr Sr Ju Su Jd Sd Jm Sm ...
        Il Ir Jl1 Sl1 Jl2 Sl2 Jr1 Sr1 Jr2 Sr2 I J S

  % factor matrix
  opts = struct('Tmax',Tmax,'skip',skip,'symm',symm,'verb',1);
  tic; F = hifde2(A,n,occ,rank_or_tol,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifde2 time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

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