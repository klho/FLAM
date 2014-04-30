% Seven-point stencil on the unit cube, Poisson equation.

function fd_cube(n,occ,rank_or_tol,skip,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 32;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 4 || isempty(skip)
    skip = 1;
  end
  if nargin < 5 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2,x3] = ndgrid((1:n)/n);
  x = [x1(:) x2(:) x3(:)]';
  N = size(x,2);
  clear x1 x2 x3

  % set up sparse matrix
  h = 1/(n + 1);
  idx = reshape(1:N,n,n,n);
  Im = idx(1:n,1:n,1:n);
  Jm = idx(1:n,1:n,1:n);
  Sm = 6/h^2*ones(size(Im));
  Il = idx(1:n-1,1:n,1:n);
  Jl = idx(2:n,  1:n,1:n);
  Sl = -1/h^2*ones(size(Il));
  Ir = idx(2:n,  1:n,1:n);
  Jr = idx(1:n-1,1:n,1:n);
  Sr = -1/h^2*ones(size(Ir));
  Iu = idx(1:n,1:n-1,1:n);
  Ju = idx(1:n,2:n  ,1:n);
  Su = -1/h^2*ones(size(Iu));
  Id = idx(1:n,2:n  ,1:n);
  Jd = idx(1:n,1:n-1,1:n);
  Sd = -1/h^2*ones(size(Id));
  If = idx(1:n,1:n,1:n-1);
  Jf = idx(1:n,1:n,2:n  );
  Sf = -1/h^2*ones(size(If));
  Ib = idx(1:n,1:n,2:n  );
  Jb = idx(1:n,1:n,1:n-1);
  Sb = -1/h^2*ones(size(Ib));
  I = [Im(:); Il(:); Ir(:); Iu(:); Id(:); If(:); Ib(:)];
  J = [Jm(:); Jl(:); Jr(:); Ju(:); Jd(:); Jf(:); Jb(:)];
  S = [Sm(:); Sl(:); Sr(:); Su(:); Sd(:); Sf(:); Sb(:)];
  A = sparse(I,J,S,N,N);
  P = zeros(N,1);
  clear idx Im Jm Sm Il Jl Sl Ir Jr Sr Iu Ju Su Id Jd Sd If Jf Sf Ib Jb Sb I J S

  % factor matrix
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = rskelf(@spget,x,occ,rank_or_tol,@pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  rskelf_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - rskelf_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = rskelf_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*rskelf_sv(F,x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@(x)(rskelf_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)

  % proxy function
  function [Kpxy,nbr] = pxyfun(x,slf,nbr,l,ctr)
    Kpxy = zeros(0,length(slf));
    snbr = sort(nbr);
    [nbr,~] = find(A(:,slf));
    nbr = nbr(ismembc(nbr,snbr));
  end

  % sparse matrix access
  function S = spget(I_,J_)
    m_ = length(I_);
    n_ = length(J_);
    [I_sort,E] = sort(I_);
    P(I_sort) = E;
    S = zeros(m_,n_);
    [I_,J_,S_] = find(A(:,J_));
    idx = ismembc(I_,I_sort);
    I_ = I_(idx);
    J_ = J_(idx);
    S_ = S_(idx);
    S(P(I_) + (J_ - 1)*m_) = S_;
  end
end