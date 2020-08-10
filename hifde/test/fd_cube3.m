% Seven-point stencil on the unit cube, constant-coefficient Helmholtz equation,
% Dirichlet boundary conditions.
%
% This is basically the 3D analogue of FD_SQUARE3.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 32)
%   - K: wavenumber (default: K = 2*PI*4)
%   - OCC: tree occupancy parameter (default: OCC = 4)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-6)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SKIP: skip parameter (default: SKIP = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'H')
%   - DOITER: whether to run unpreconditioned MINRES (default: DOITER = 1)
%   - DIAGMODE: diagonal extraction mode - 0: skip; 1: matrix unfolding; 2:
%       sparse apply/solves (default: DIAGMODE = 0)

function fd_cube3(n,k,occ,rank_or_tol,Tmax,skip,symm,doiter,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 32; end
  if nargin < 2 || isempty(k), k = 2*pi*4; end
  if nargin < 3 || isempty(occ), occ = 4; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(skip), skip = 2; end
  if nargin < 7 || isempty(symm), symm = 'h'; end
  if nargin < 8 || isempty(doiter), doiter = 1; end
  if nargin < 9 || isempty(diagmode), diagmode = 0; end

  % initialize
  N = (n - 1)^3;  % total number of grid points
  h = 1/n;        % mesh width

  % set up sparse matrix
  idx = zeros(n+1,n+1,n+1);  % index mapping to each point, including "ghosts"
  idx(2:n,2:n,2:n) = reshape(1:N,n-1,n-1,n-1);
  mid = 2:n;    % "middle" indices -- interaction with self
  lft = 1:n-1;  % "left"   indices -- interaction with one below
  rgt = 3:n+1;  % "right"  indices -- interaction with one above
  I = idx(mid,mid,mid); e = ones(size(I));
  % interactions with ...
  Jl = idx(lft,mid,mid); Sl = -e;  % ... left
  Jr = idx(rgt,mid,mid); Sr = -e;  % ... right
  Ju = idx(mid,lft,mid); Su = -e;  % ... up
  Jd = idx(mid,rgt,mid); Sd = -e;  % ... down
  Jf = idx(mid,mid,lft); Sf = -e;  % ... front
  Jb = idx(mid,mid,rgt); Sb = -e;  % ... back
  Jm = idx(mid,mid,mid);           % ... middle (self)
  Sm = -(Sl + Sr + Sd + Su + Sb + Sf) - h^2*k^2*e;
  % combine all interactions
  I = [ I(:);  I(:);  I(:);  I(:);  I(:);  I(:);  I(:)];
  J = [Jl(:); Jr(:); Ju(:); Jd(:); Jf(:); Jb(:); Jm(:)];
  S = [Sl(:); Sr(:); Su(:); Sd(:); Sf(:); Sb(:); Sm(:)];
  % remove ghost interactions
  idx = find(J > 0); I = I(idx); J = J(idx); S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Jl Sl Jr Sr Ju Su Jd Sd Jf Sf Jb Sb Im Sm I J S

  % factor matrix
  opts = struct('Tmax',Tmax,'skip',skip,'symm',symm,'verb',1);
  tic; F = hifde3(A,n,occ,rank_or_tol,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifde3 time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

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

  B = A*X;
  do_minres = 1;
  if isoctave()
    warning('No MINRES in Octave; using GMRES.')
    do_minres = 0;
  end

  % run unpreconditioned MINRES
  if do_minres
    iter = nan;
    if doiter, [~,~,~,iter] = minres(@(x)(A*x),B,1e-12,128); end
  else
    iter(1:2) = nan;
    if doiter, [~,~,~,iter] = gmres(@(x)(A*x),B,32,1e-12,32); end
    iter = (iter(1) + 1)*iter(2);  % total iterations
  end

  % run preconditioned MINRES
  tic
  if do_minres
    [Y,~,~,piter] = minres(@(x)(A*x),B,1e-12,32,@(x)hifde_sv(F,x));
  else
    [Y,~,~,piter] = gmres(@(x)(A*x),B,32,1e-12,32,@(x)hifde_sv(F,x));
    piter = (piter(1) + 1)*piter(2);  % total iterations
  end
  t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - A*Y)/norm(B);
  if do_minres, fprintf('minres:\n')
  else,         fprintf('gmres:\n')
  end
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
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