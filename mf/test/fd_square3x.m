% Five-point stencil on the unit square, constant-coefficient Helmholtz,
% Dirichlet boundary conditions.

function fd_square3x(n,k,occ,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(k)
    k = 2*pi*8;
  end
  if nargin < 3 || isempty(occ)
    occ = 64;
  end
  if nargin < 4 || isempty(symm)
    symm = 'h';
  end

  % initialize
  [x1,x2] = ndgrid((1:n-1)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2

  % set up indices
  idx = zeros(n+1,n+1);
  idx(2:n,2:n) = reshape(1:N,n-1,n-1);
  mid = 2:n;
  lft = 1:n-1;
  rgt = 3:n+1;

  % interactions with left node
  Il = idx(mid,mid);
  Jl = idx(lft,mid);
  Sl = -1/h^2*ones(size(Il));

  % interactions with right node
  Ir = idx(mid,mid);
  Jr = idx(rgt,mid);
  Sr = -1/h^2*ones(size(Ir));

  % interactions with bottom node
  Id = idx(mid,mid);
  Jd = idx(mid,lft);
  Sd = -1/h^2*ones(size(Id));

  % interactions with top node
  Iu = idx(mid,mid);
  Ju = idx(mid,rgt);
  Su = -1/h^2*ones(size(Iu));

  % interactions with self
  Im = idx(mid,mid);
  Jm = idx(mid,mid);
  Sm = -(Sl + Sr + Sd + Su) - k^2*ones(size(Im));

  % form sparse matrix
  I = [Il(:); Ir(:); Id(:); Iu(:); Im(:)];
  J = [Jl(:); Jr(:); Jd(:); Ju(:); Jm(:)];
  S = [Sl(:); Sr(:); Sd(:); Su(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il Jl Sl Ir Jr Sr Id Jd Sd Iu Ju Su Im Jm Sm I J S

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

  % run unpreconditioned GMRES
  [~,~,~,iter] = gmres(@(x)(A*x),X,[],1e-12,128);

  % run preconditioned GMRES
  tic
  [Z,~,~,piter] = gmres(@(x)(A*x),X,[],1e-12,32,@(x)(mf_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('gmres: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter(2),iter(2),t)

  % compute log-determinant
  tic
  ld = mf_logdet(F);
  t = toc;
  fprintf('logdet: %22.16e / %10.4e (s)\n',ld,t)
end