% Three-point stencil on the unit line, constant-coefficient Poisson, Dirichlet
% boundary conditions.

function fd_line1x_diag(n,occ,symm,spdiag)

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
  if nargin < 4 || isempty(spdiag)
    spdiag = 0;
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

  % prepare for diagonal extracation
  opts = struct('verb',1);
  r = randperm(N);
  m = min(N,128);
  r = r(1:m);
  X = zeros(N,m);
  for i = 1:m
    X(r(i),i) = 1;
  end
  E = zeros(m,1);

  % extract diagonal
  if spdiag
    tic
    D = mf_spdiag(F);
    t1 = toc;
  else
    D = mf_diag(F,0,opts);
  end
  Y = mf_mv(F,X);
  for i = 1:m
    E(i) = Y(r(i),i);
  end
  e1 = norm(D(r) - E)/norm(E);
  if spdiag
    fprintf('spdiag_mv: %10.4e / %10.4e (s)\n',e1,t1)
  end

  % extract diagonal of inverse
  if spdiag
    tic
    D = mf_spdiag(F,1);
    t2 = toc;
  else
    D = mf_diag(F,1,opts);
  end
  Y = mf_sv(F,X);
  for i = 1:m
    E(i) = Y(r(i),i);
  end
  e2 = norm(D(r) - E)/norm(E);
  if spdiag
    fprintf('spdiag_sv: %10.4e / %10.4e (s)\n',e2,t2)
  end

  % print summary
  if ~spdiag
    fprintf([repmat('-',1,80) '\n'])
    fprintf('diag: %10.4e / %10.4e\n',e1,e2)
  end
end