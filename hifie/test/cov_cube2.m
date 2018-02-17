% Matern covariance function on the unit cube.

function cov_cube2(n,occ,p,rank_or_tol,skip,symm,noise,scale,spdiag)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 32;
  end
  if nargin < 2 || isempty(occ)
    occ = 512;
  end
  if nargin < 3 || isempty(p)
    p = 16;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-3;
  end
  if nargin < 5 || isempty(skip)
    skip = 0;
  end
  if nargin < 6 || isempty(symm)
    symm = 'p';
  end
  if nargin < 7 || isempty(noise)
    noise = 1e-2;
  end
  if nargin < 8 || isempty(scale)
    scale = 100;
  end
  if nargin < 9 || isempty(spdiag)
    spdiag = 0;
  end

  % initialize
  [x1,x2,x3] = ndgrid((1:n)/n);
  x = [x1(:) x2(:) x3(:)]';
  N = size(x,2);
  proxy_ = randn(3,p);
  proxy_ = bsxfun(@rdivide,proxy_,sqrt(sum(proxy_.^2)));
  proxy = [];
  for r = linspace(1.5,2.5,p)
    proxy = [proxy r*proxy_];
  end
  clear x1 x2 x3

  % factor matrix
  Afun = @(i,j)Afun2(i,j,x,noise);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun2(x,slf,nbr,l,ctr,proxy);
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = hifie3(Afun,x,occ,rank_or_tol,pxyfun,opts);
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
  hifie_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(mv(x) - hifie_mv(F,x)),[],[],1);
  e = e/snorm(N,mv,[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  hifie_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - mv(hifie_sv(F,x))),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic
    hifie_cholmv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(hifie_mv(F,x) ...
                           - hifie_cholmv(F,hifie_cholmv(F,x,'c'))),[],[],1);
    e = e/snorm(N,@(x)(hifie_mv(F,x)),[],[],1);
    fprintf('cholmv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic
    hifie_cholsv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(hifie_sv(F,x) ...
                           - hifie_cholsv(F,hifie_cholsv(F,x),'c')),[],[],1);
    e = e/snorm(N,@(x)(hifie_sv(F,x)),[],[],1);
    fprintf('cholsv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)
  end

  % compute log-determinant
  tic
  ld = hifie_logdet(F);
  t = toc;
  fprintf('logdet: %22.16e / %10.4e (s)\n',ld,t)

  % prepare for diagonal extracation
  opts = struct('verb',1);
  r = randperm(N);
  m = min(N,128);
  r = r(1:m);
  X = zeros(N,m);
  for i = 1:m
    X(r(i),i) = 1;
  end
  E = zeros(m,1);

  % extract diagonal
  if spdiag
    tic
    D = hifie_spdiag(F);
    t1 = toc;
  else
    D = hifie_diag(F,0,opts);
  end
  Y = hifie_mv(F,X);
  for i = 1:m
    E(i) = Y(r(i),i);
  end
  e1 = norm(D(r) - E)/norm(E);
  if spdiag
    fprintf('spdiag_mv: %10.4e / %10.4e (s)\n',e1,t1)
  end

  % extract diagonal of inverse
  if spdiag
    tic
    D = hifie_spdiag(F,1);
    t2 = toc;
  else
    D = hifie_diag(F,1,opts);
  end
  Y = hifie_sv(F,X);
  for i = 1:m
    E(i) = Y(r(i),i);
  end
  e2 = norm(D(r) - E)/norm(E);
  if spdiag
    fprintf('spdiag_sv: %10.4e / %10.4e (s)\n',e2,t2)
  end

  % print summary
  if ~spdiag
    fprintf([repmat('-',1,80) '\n'])
    fprintf('diag: %10.4e / %10.4e\n',e1,e2)
  end

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dz = bsxfun(@minus,x(3,:)',y(3,:));
    dr = scale*sqrt(dx.^2 + dy.^2 + dz.^2);
    K = (1 + sqrt(3)*dr).*exp(-sqrt(3)*dr);
  end
end

% matrix entries
function A = Afun2(i,j,x,noise)
  A = Kfun(x(:,i),x(:,j));
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = A(idx) + noise^2;
end

% proxy function
function [Kpxy,nbr] = pxyfun2(x,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  Kpxy = Kfun(pxy,x(:,slf));
  dx = x(1,nbr) - ctr(1);
  dy = x(2,nbr) - ctr(2);
  dz = x(3,nbr) - ctr(3);
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