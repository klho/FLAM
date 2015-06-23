% Three-point stencil on the unit line, constant-coefficient Poisson, Dirichlet
% boundary conditions.

function fd_line1x(n,occ,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(symm)
    symm = 'p';
  end

  % initialize
  x = (1:n-1)/n;
  N = size(x,2);
  h = 1/n;

  % set up indices
  idx = zeros(n+1,1);
  idx(2:n) = 1:N;
  mid = 2:n;
  lft = 1:n-1;
  rgt = 3:n+1;

  % interactions with left node
  Il = idx(mid);
  Jl = idx(lft);
  Sl = -1/h^2*ones(size(Il));

  % interactions with right node
  Ir = idx(mid);
  Jr = idx(rgt);
  Sr = -1/h^2*ones(size(Ir));

  % interactions with self
  Im = idx(mid);
  Jm = idx(mid);
  Sm = -(Sl + Sr);

  % form sparse matrix
  I = [Il(:); Ir(:); Im(:)];
  J = [Jl(:); Jr(:); Jm(:)];
  S = [Sl(:); Sr(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il Jl Sl Ir Jr Sr Im Jm Sm I J S

  % factor matrix
  opts = struct('symm',symm,'verb',1);
  F = mfx(A,x,occ,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  mf_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - mf_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = mf_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*mf_sv(F,x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic
    mf_cholmv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(mf_mv(F,x) ...
                           - mf_cholmv(F,mf_cholmv(F,x,'c'))),[],[],1);
    e = e/snorm(N,@(x)(mf_mv(F,x)),[],[],1);
    fprintf('cholmv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic
    mf_cholsv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(mf_sv(F,x) ...
                           - mf_cholsv(F,mf_cholsv(F,x),'c')),[],[],1);
    e = e/snorm(N,@(x)(mf_sv(F,x)),[],[],1);
    fprintf('cholsv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)
  end

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@(x)(mf_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)

  % compute log-determinant
  tic
  ld = mf_logdet(F);
  t = toc;
  fprintf('logdet: %22.16e / %10.4e (s)\n',ld,t)
end