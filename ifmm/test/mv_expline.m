% Exponentially graded line, Laplace sources.

function mv_expline(n,occ,p,rank_or_tol,near,store,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 64;
  end
  if nargin < 2 || isempty(occ)
    occ = 32;
  end
  if nargin < 3 || isempty(p)
    p = 8;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 5 || isempty(near)
    near = 0;
  end
  if nargin < 6 || isempty(store)
    store = 'n';
  end
  if nargin < 7 || isempty(symm)
    symm = 's';
  end

  % initialize
  x = 2.^(-(1:n));
  N = size(x,2);
  proxy = 1.5 + ((1:p) - 1)/p;
  proxy = [proxy -proxy];

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'symm',symm,'verb',1);
  F = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % set up accuracy tests
  A = Afun(1:N,1:N);

  % test matrix apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  ifmm_mv(F,X,Afun,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  Y = ifmm_mv(F,X,Afun,'n');
  Z = A*X;
  e = norm(Z - Y)/norm(Z);
  fprintf('mv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix adjoint apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  ifmm_mv(F,X,Afun,'c');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  Y = ifmm_mv(F,X,Afun,'c');
  Z = A'*X;
  e = norm(Z - Y)/norm(Z);
  fprintf('mva: %10.4e / %10.4e (s)\n',e,t)

  % kernel function
  function K = Kfun(x,y)
    dr = abs(bsxfun(@minus,x',y));
    K = dr;
  end
end

% matrix entries
function A = Afun2(i,j,x)
  A = Kfun(x(:,i),x(:,j));
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy);
    dist = cx(:,nbr) - ctr;
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf));
    dist = rx(:,nbr) - ctr;
  end
  nbr = nbr(dist/l < 1.5);
end