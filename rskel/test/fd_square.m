% Five-point stencil on the unit square, Poisson equation.

function fd_square(n,occ,rank_or_tol,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
  end
  if nargin < 3 || isempty(rank_or_tol)
    rank_or_tol = 1e-9;
  end
  if nargin < 4 || isempty(symm)
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

  % compress matrix
  Afun = @(i,j)Afun2(i,j,A,N);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,A);
  opts = struct('symm',symm,'verb',1);
  F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % factor extended sparsification
  tic
  S = rskel_xsp(F);
  t = toc;
  w = whos('S');
  fprintf('xsp: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6);
  dolu = strcmpi(F.symm,'n');
  if ~dolu && isoctave
    dolu = 1;
    S = S + tril(S,-1)';
  end
  FS = struct('lu',dolu);
  tic
  if dolu
    [FS.L,FS.U] = lu(S);
  else
    [FS.L,FS.D,FS.P] = ldl(S);
  end
  t = toc;
  if dolu
    w = whos('FS.L');
    spmem = w.bytes;
    w = whos('FS.U');
    spmem = (spmem + w.bytes)/1e6;
  else
    w = whos('FS.L');
    spmem = w.bytes;
    w = whos('FS.D');
    spmem = (spmem + w.bytes)/1e6;
  end
  fprintf('lu/ldl: %10.4e (s) / %6.2f (MB)\n',t,spmem)
  sv = @(x)sv2(FS,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  rskel_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - rskel_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = sv(X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*sv(x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,sv);
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
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,A)
  Kpxy = zeros(0,length(slf));
  if strcmpi(rc,'r')
    Kpxy = Kpxy';
  end
  snbr = sort(nbr);
  [nbr,~] = find(A(:,slf));
  nbr = nbr(ismemb(nbr,snbr));
end

% sparse LU solve
function Y = sv2(F,X)
  N = size(X,1);
  X = [X; zeros(size(F.L,1)-N,size(X,2))];
  if F.lu
    Y = F.U\(F.L\X);
  else
    Y = F.P*(F.L'\(F.D\(F.L\(F.P'*X))));
  end
  Y = Y(1:N,:);
end