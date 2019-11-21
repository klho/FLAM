% Seven-point stencil on the unit cube, constant-coefficient Helmholtz equation,
% Dirichlet boundary conditions.
%
% This is the same as FD_CUBE3 but using HIFDE3X.

function fd_cube3x(n,k,occ,rank_or_tol,skip,symm,doiter,diagmode)

  % set default parameters
  if nargin < 1 || isempty(n), n = 32; end  % number of points + 1 in each dim
  if nargin < 2 || isempty(k), k = 2*pi*4; end  % wavenumber
  if nargin < 3 || isempty(occ), occ = 64; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 5 || isempty(skip), skip = 2; end
  if nargin < 6 || isempty(symm), symm = 'h'; end  % positive definite
  if nargin < 7 || isempty(doiter), doiter = 1; end  % unpreconditioned GMRES?
  if nargin < 8 || isempty(diagmode), diagmode = 0; end  % diag extraction mode:
  % 0 - skip; 1 - matrix unfolding; 2 - sparse apply/solves

  % initialize
  [x1,x2,x3] = ndgrid((1:n-1)/n); x = [x1(:) x2(:) x3(:)]';  % grid points
  clear x1 x2 x3
  N = size(x,2);
  h = 1/n;  % mesh width

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
  opts = struct('skip',skip,'symm',symm,'verb',1);
  tic; F = hifde3x(A,x,occ,rank_or_tol,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('hifde3x time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

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

  % run unpreconditioned GMRES
  B = A*X;
  iter(2) = nan;
  if doiter, [~,~,~,iter] = gmres(@(x)(A*x),B,[],1e-12,128); end

  % run preconditioned GMRES
  tic;
  [Y,~,~,piter] = gmres(@(x)(A*x),B,[],1e-12,32,@(x)hifde_sv(F,x));
  t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - A*Y)/norm(B);
  fprintf('gmres:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',piter(2),iter(2))

  % compute log-determinant
  tic
  ld = hifde_logdet(F);
  t = toc;
  fprintf('hifde_logdet: %22.16e / %10.4e (s)\n',ld,t)

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