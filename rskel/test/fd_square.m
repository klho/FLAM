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
    symm = 's';
  end

  % initialize
  [x1,x2] = ndgrid((1:n)/(n + 1));
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
  clear idx Im Jm Sm Il Jl Sl Ir Jr Sr Iu Ju Su Id Jd Sd I J S

  % convert to CSC format
  [Ai,Ax,P] = spcsc(A);

  % compress matrix
  opts = struct('symm',symm,'verb',1);
  F = rskel(@(i,j)(spsmat(Ai,Ax,P,i,j)),x,x,occ,rank_or_tol,@pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % factor extended sparsification
  tic
  S = rskel_xsp(F);
  t = toc;
  w = whos('S');
  fprintf('xsp: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6);
  tic
  if strcmp(F.symm,'n')
    [L,U] = lu(S);
  elseif strcmp(F.symm,'s') || strcmp(F.symm,'h')
    [L,D,P] = ldl(S);
  end
  t = toc;
  if strcmp(F.symm,'n')
    w = whos('L');
    spmem = w.bytes;
    w = whos('U');
    spmem = (spmem + w.bytes)/1e6;
  elseif strcmp(F.symm,'s') || strcmp(F.symm,'h')
    w = whos('L');
    spmem = w.bytes;
    w = whos('D');
    spmem = (spmem + w.bytes)/1e6;
  end
  fprintf('lu/ldl: %10.4e (s) / %6.2f (MB)\n',t,spmem)

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
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@sv);
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)

  % proxy function
  function [Kpxy,nbr] = pxyfun(rc,rx,cx,slf,nbr,l,ctr)
    Kpxy = zeros(0,length(slf));
    if strcmp(rc,'r')
      Kpxy = Kpxy';
    end
    snbr = sort(nbr);
    nbr = vertcat(Ai{slf});
    nbr = nbr(ismembc(nbr,snbr));
  end

  % sparse LU solve
  function Y = sv(X)
    X = [X; zeros(size(S,1)-N,size(X,2))];
    if strcmp(F.symm,'n')
      Y = U\(L\X);
    elseif strcmp(F.symm,'s') || strcmp(F.symm,'h')
      Y = P*(L'\(D\(L\(P'*X))));
    end
    Y = Y(1:N,:);
  end
end