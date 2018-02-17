% Unit square, Helmholtz sources.

function mv_square2(m,n,k,occ,p,rank_or_tol,near,store)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 16384;
  end
  if nargin < 2 || isempty(n)
    n = 8192;
  end
  if nargin < 3 || isempty(k)
    k = 2*pi*8;
  end
  if nargin < 4 || isempty(occ)
    occ = 128;
  end
  if nargin < 5 || isempty(p)
    p = 64;
  end
  if nargin < 6 || isempty(rank_or_tol)
    rank_or_tol = 1e-9;
  end
  if nargin < 7 || isempty(near)
    near = 0;
  end
  if nargin < 8 || isempty(store)
    store = 'n';
  end

  % initialize
  rx = rand(2,m);
  cx = rand(2,n);
  M = size(rx,2);
  N = size(cx,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];

  % compress matrix
  Afun = @(i,j)Afun2(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'verb',1);
  F = ifmm(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test matrix apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  ifmm_mv(F,X,Afun,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(M);
  r = r(1:min(M,128));
  A = Afun(r,1:N);
  Y = ifmm_mv(F,X,Afun,'n');
  Z = A*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix adjoint apply accuracy
  X = rand(M,1);
  X = X/norm(X);
  tic
  ifmm_mv(F,X,Afun,'c');
  t = toc;
  X = rand(M,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  A = Afun(1:M,r);
  Y = ifmm_mv(F,X,Afun,'c');
  Z = A'*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mva: %10.4e / %10.4e (s)\n',e,t)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    K = 0.25i*besselh(0,1,k*sqrt(dx.^2 + dy.^2));
  end
end

% matrix entries
function A = Afun2(i,j,rx,cx)
  A = Kfun(rx(:,i),cx(:,j));
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf));
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end