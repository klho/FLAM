% Seven-point stencil on the unit cube, constant-coefficient Poisson, Dirichlet
% boundary conditions.

function fd_cube1x_diag(n,occ,symm,spdiag)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 32;
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
  [x1,x2,x3] = ndgrid((1:n-1)/n);
  x = [x1(:) x2(:) x3(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2 x3

  % set up indices
  idx = zeros(n+1,n+1,n+1);
  idx(2:n,2:n,2:n) = reshape(1:N,n-1,n-1,n-1);
  mid = 2:n;
  lft = 1:n-1;
  rgt = 3:n+1;

  % interactions with left node
  Il = idx(mid,mid,mid);
  Jl = idx(lft,mid,mid);
  Sl = -1/h^2*ones(size(Il));

  % interactions with right node
  Ir = idx(mid,mid,mid);
  Jr = idx(rgt,mid,mid);
  Sr = -1/h^2*ones(size(Ir));

  % interactions with bottom node
  Id = idx(mid,mid,mid);
  Jd = idx(mid,lft,mid);
  Sd = -1/h^2*ones(size(Id));

  % interactions with top node
  Iu = idx(mid,mid,mid);
  Ju = idx(mid,rgt,mid);
  Su = -1/h^2*ones(size(Iu));

  % interactions with back node
  Ib = idx(mid,mid,mid);
  Jb = idx(mid,mid,lft);
  Sb = -1/h^2*ones(size(Ib));

  % interactions with front node
  If = idx(mid,mid,mid);
  Jf = idx(mid,mid,rgt);
  Sf = -1/h^2*ones(size(If));

  % interactions with self
  Im = idx(mid,mid,mid);
  Jm = idx(mid,mid,mid);
  Sm = -(Sl + Sr + Sd + Su + Sb + Sf);

  % form sparse matrix
  I = [Il(:); Ir(:); Id(:); Iu(:); Ib(:); If(:); Im(:)];
  J = [Jl(:); Jr(:); Jd(:); Ju(:); Jb(:); Jf(:); Jm(:)];
  S = [Sl(:); Sr(:); Sd(:); Su(:); Sb(:); Sf(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il Jl Sl Ir Jr Sr Id Jd Sd Iu Ju Su Ib Jb Sb If Jf Sf Im Jm Sm I J S

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