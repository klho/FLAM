% Second-kind integral equation on the unit circle, Laplace double-layer.

function ie_circle(n,occ,p,rank_or_tol,near,store,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
  end
  if nargin < 3 || isempty(p)
    p = 64;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
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
  theta = (1:n)*2*pi/n;
  x = [cos(theta); sin(theta)];
  N = size(x,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('near',near,'store',store,'symm',symm,'verb',1);
  F = ifmm(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % set up FFT multiplication
  G = fft(Afun(1:N,1));
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

  % generate field due to exterior sources
  m = 16;
  theta = (1:m)*2*pi/m;
  src = 2*[cos(theta); sin(theta)];
  q = rand(m,1);
  B = Kfun(x,src,'s')*q;

  % solve for surface density
  tic
  [X,~,~,iter] = gmres(@(x)(ifmm_mv(F,x,Afun)),B,[],1e-12,32);
  t = toc;
  e = norm(B - mv(X))/norm(B);
  fprintf('gmres: %10.4e / %4d / %10.4e (s)\n',e,iter(2),t)

  % evaluate field at interior targets
  trg = 0.5*[cos(theta); sin(theta)];
  Y = Kfun(trg,x,'d')*(2*pi/N)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  e = norm(Z - Y)/norm(Z);
  fprintf('pde: %10.4e\n',e)

  % kernel function
  function K = Kfun(x,y,lp)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2 + dy.^2);
    if strcmpi(lp,'s')
      K = -1/(2*pi)*log(sqrt(dr));
    elseif strcmpi(lp,'d')
      rdotn = bsxfun(@times,dx,y(1,:)) + bsxfun(@times,dy,y(2,:));
      K = 1/(2*pi).*rdotn./dr.^2;
    end
  end
end

% matrix entries
function A = Afun2(i,j,x)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j),'d')*(2*pi/N);
  [I,J] = ndgrid(i,j);
  A(I == J) = -0.5*(1 + 1/N);
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  N = size(rx,2);
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy,'s')*(2*pi/N);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf),'s')*(2*pi/N);
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv2(F,x)
  y = ifft(F.*fft(x));
end