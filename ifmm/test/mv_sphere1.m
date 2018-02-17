% Unit sphere, Laplace sources.

function mv_sphere1(m,n,occ,p,rank_or_tol,near,store)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 16384;
  end
  if nargin < 2 || isempty(n)
    n = 8192;
  end
  if nargin < 3 || isempty(occ)
    occ = 1024;
  end
  if nargin < 4 || isempty(p)
    p = 512;
  end
  if nargin < 5 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 6 || isempty(near)
    near = 0;
  end
  if nargin < 7 || isempty(store)
    store = 'n';
  end

  % initialize
  rx = randn(3,m);
  rx = bsxfun(@rdivide,rx,sqrt(sum(rx.^2)));
  cx = randn(3,n);
  cx = bsxfun(@rdivide,cx,sqrt(sum(cx.^2)));
  M = size(rx,2);
  N = size(cx,2);
  proxy = randn(3,p);
  proxy = 1.5*bsxfun(@rdivide,proxy,sqrt(sum(proxy.^2)));

  % compress matrix
  Afun = @(i,j)Afun2(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'verb',1);
  F = ifmm(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

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
    dz = bsxfun(@minus,x(3,:)',y(3,:));
    K = 1/(4*pi)./sqrt(dx.^2 + dy.^2 + dz.^2);
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
    dz = cx(3,nbr) - ctr(3);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf));
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
    dz = rx(3,nbr) - ctr(3);
  end
  dist = sqrt(dx.^2 + dy.^2 + dz.^2);
  nbr = nbr(dist/l < 1.5);
end