% Five-point stencil on the unit square, Poisson equation.

function fd_square(n,occ,rank_or_tol,skip,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(rank_or_tol)
    rank_or_tol = 1e-9;
  end
  if nargin < 4 || isempty(skip)
    skip = 1;
  end
  if nargin < 5 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2] = ndgrid((1:n)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  clear x1 x2

  % set up sparse matrix
  h = 1/(n + 1);
  idx = reshape(1:N,n,n);
  Im = idx(1:n,1:n);
  Jm = idx(1:n,1:n);
  Sm = 4/h^2*ones(size(Im));
  Il = idx(1:n-1,1:n);
  Jl = idx(2:n,  1:n);
  Sl = -1/h^2*ones(size(Il));
  Ir = idx(2:n,  1:n);
  Jr = idx(1:n-1,1:n);
  Sr = -1/h^2*ones(size(Ir));
  Iu = idx(1:n,1:n-1);
  Ju = idx(1:n,2:n  );
  Su = -1/h^2*ones(size(Iu));
  Id = idx(1:n,2:n  );
  Jd = idx(1:n,1:n-1);
  Sd = -1/h^2*ones(size(Id));
  I = [Im(:); Il(:); Ir(:); Iu(:); Id(:)];
  J = [Jm(:); Jl(:); Jr(:); Ju(:); Jd(:)];
  S = [Sm(:); Sl(:); Sr(:); Su(:); Sd(:)];
  A = sparse(I,J,S,N,N);
  P = zeros(N,1);
  clear idx Im Jm Sm Il Jl Sl Ir Jr Sr Iu Ju Su Id Jd Sd I J S

  % factor matrix
  Afun = @(i,j)Afun2(i,j,A,N);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun2(x,slf,nbr,l,ctr,A);
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = hifie2(Afun,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  hifie_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - hifie_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = hifie_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*hifie_sv(F,x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@(x)(hifie_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)
end

% matrix entries
function X = Afun2(i,j,A,N)
  persistent P
  if isempty(P)
    P = zeros(N,1);
  end
  m = length(i);
  n = length(j);
  [I_sort,E] = sort(i);
  P(I_sort) = E;
  X = zeros(m,n);
  [I,J,S] = find(A(:,j));
  idx = ismemb(I,I_sort);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  X(P(I) + (J - 1)*m) = S;
end

% proxy function
function [Kpxy,nbr] = pxyfun2(x,slf,nbr,l,ctr,A)
  Kpxy = zeros(0,length(slf));
  snbr = sort(nbr);
  [nbr,~] = find(A(:,slf));
  nbr = nbr(ismemb(nbr,snbr));
end