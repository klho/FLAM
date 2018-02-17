% Unit line, Laplace sources.

function mv_line(n,occ,p,rank_or_tol,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
  end
  if nargin < 3 || isempty(p)
    p = 8;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 5 || isempty(symm)
    symm = 'n';
  end

  % initialize
  x = rand(1,n);
  N = size(x,2);
  proxy = 1.5 + ((1:p) - 1)/p;
  proxy = [proxy -proxy];

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('symm',symm,'verb',1);
  F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test matrix apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskel_mv(F,X,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  A = Afun(r,1:N);
  Y = rskel_mv(F,X,'n');
  Z = A*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix adjoint apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskel_mv(F,X,'c');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  A = Afun(1:N,r);
  Y = rskel_mv(F,X,'c');
  Z = A'*X;
  e = norm(Z - Y(r,:))/norm(Z);
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
  N = size(rx,2);
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy)/N;
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf))/N;
  end
end