% Second-kind integral equation on the unit cube, Laplace single-layer.

function ie_cube(n,occ,p,rank_or_tol,near,store,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 32;
  end
  if nargin < 2 || isempty(occ)
    occ = 1024;
  end
  if nargin < 3 || isempty(p)
    p = 512;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 5 || isempty(near)
    near = 0;
  end
  if nargin < 6 || isempty(store)
    store = 'a';
  end
  if nargin < 7 || isempty(symm)
    symm = 's';
  end

  % initialize
  [x1,x2,x3] = ndgrid((1:n)/n);
  x = [x1(:) x2(:) x3(:)]';
  N = size(x,2);
  proxy = randn(3,p);
  proxy = 1.5*bsxfun(@rdivide,proxy,sqrt(sum(proxy.^2)));
  clear x1 x2 x3

  % compute diagonal quadratures
  h = 1/n;
  intgrl = 8*triplequad(@(x,y,z)(1/(4*pi)./sqrt(x.^2 + y.^2 + z.^2)), ...
                        0,h/2,0,h/2,0,h/2);

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x,intgrl);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'symm',symm,'verb',1);
  F = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % set up FFT multiplication
  a = reshape(Afun(1:N,1),n,n,n);
  B = zeros(2*n-1,2*n-1,2*n-1);
  B(  1:n  ,  1:n  ,  1:n  ) = a;
  B(  1:n  ,  1:n  ,n+1:end) = a(:,:,2:n);
  B(  1:n  ,n+1:end,  1:n  ) = a(:,2:n,:);
  B(  1:n  ,n+1:end,n+1:end) = a(:,2:n,2:n);
  B(n+1:end,  1:n  ,  1:n  ) = a(2:n,:,:);
  B(n+1:end,  1:n  ,n+1:end) = a(2:n,:,2:n);
  B(n+1:end,n+1:end,  1:n  ) = a(2:n,2:n,:);
  B(n+1:end,n+1:end,n+1:end) = a(2:n,2:n,2:n);
  B(:,:,n+1:end) = flipdim(B(:,:,n+1:end),3);
  B(:,n+1:end,:) = flipdim(B(:,n+1:end,:),2);
  B(n+1:end,:,:) = flipdim(B(n+1:end,:,:),1);
  G = fftn(B);
  mv = @(x)mv2(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  ifmm_mv(F,X,Afun);
  t = toc;
  [e,niter] = snorm(N,@(x)(mv(x) - ifmm_mv(F,x,Afun)),[],[],1);
  e = e/snorm(N,mv,[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run GMRES
  tic
  [Y,~,~,iter] = gmres(@(x)(ifmm_mv(F,x,Afun)),X,[],1e-12,32);
  t = toc;
  e = norm(X - mv(Y))/norm(X);
  fprintf('gmres: %10.4e / %4d / %10.4e (s)\n',e,iter(2),t)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dz = bsxfun(@minus,x(3,:)',y(3,:));
    K = -1/(4*pi)*log(sqrt(dx.^2 + dy.^2 + dz.^2));
  end
end

% matrix entries
function A = Afun2(i,j,x,intgrl)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j))/N;
  [I,J] = ndgrid(i,j);
  A(I == J) = 1 + intgrl;
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  N = size(rx,2);
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy)/N;
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
    dz = cx(3,nbr) - ctr(3);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf))/N;
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
    dz = rx(3,nbr) - ctr(3);
  end
  dist = sqrt(dx.^2 + dy.^2 + dz.^2);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv2(F,x)
  N = length(x);
  n = round(N^(1/3));
  y = ifftn(F.*fftn(reshape(x,n,n,n),[2*n-1 2*n-1 2*n-1]));
  y = reshape(y(1:n,1:n,1:n),N,1);
end