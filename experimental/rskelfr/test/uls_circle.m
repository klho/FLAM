% Underdetermined least squares on the unit circle, Laplace sources.

function uls_circle(m,n,delta,occ,p,rank_or_tol,rdpiv,store)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 8192;
  end
  if nargin < 2 || isempty(n)
    n = 16384;
  end
  if nargin < 3 || isempty(delta)
    delta = 1e-12;
  end
  if nargin < 4 || isempty(occ)
    occ = 128;
  end
  if nargin < 5 || isempty(p)
    p = 64;
  end
  if nargin < 6 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 7 || isempty(rdpiv)
    rdpiv = 'l';
  end
  if nargin < 8 || isempty(store)
    store = 'a';
  end

  % initialize
  theta = (1:m)*2*pi/m;
  rx = (1 + delta)*[cos(theta); sin(theta)];
  theta = (1:n)*2*pi/n;
  cx = [cos(theta); sin(theta)];
  M = size(rx,2);
  N = size(cx,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];

  % compress matrix using RSKELFR
  Afun = @(i,j)Afun2(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('rdpiv',rdpiv,'verb',1);
  F = rskelfr(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % compress matrix using IFMM
  opts = struct('store',store,'verb',1);
  G = ifmm(Afun,rx,cx,2*occ,1e-15,pxyfun,opts);
  w = whos('G');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  rskelfr_mv(F,X);
  t1 = toc;
  tic
  ifmm_mv(G,X,Afun);
  t2 = toc;
  [e,niter] = snorm(N,@(x)(ifmm_mv(G,x,Afun,'n') - rskelfr_mv(F,x,'n')), ...
                      @(x)(ifmm_mv(G,x,Afun,'c') - rskelfr_mv(F,x,'c')));
  e = e/snorm(N,@(x)(ifmm_mv(G,x,Afun,'n')),@(x)(ifmm_mv(G,x,Afun,'c')));
  fprintf('mv: %10.4e / %4d / %10.4e (s) / %10.4e (s)\n',e,niter,t1,t2)

  % test weak pseudoinverse apply accuracy
  X = rand(M,1);
  X = X/norm(X);
  tic
  rskelfr_sv(F,X);
  t = toc;

  % M >= N (overdetermined): NORM(I - PINV(F)*A)
  if M >= N
    [e,niter] = snorm(N,@(x)(x - rskelfr_sv(F,ifmm_mv(G,x,Afun,'n'),'n')),
                        @(x)(x - ifmm_mv(G,rskelfr_sv(F,x,'c'),Afun,'c')));

  % M < N (underdetermined): NORM(I - A*PINV(F))
  else
    [e,niter] = snorm(M,@(x)(x - ifmm_mv(G,rskelfr_sv(F,x,'n'),Afun,'n')),
                        @(x)(x - rskelfr_sv(F,ifmm_mv(G,x,Afun,'c'),'c')));
  end
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % example application with simple estimates for residual/norm factors
  m = 16;
  B = rand(M,m);
  X = rskelfr_sv(F,B);
  C = ifmm_mv(G,X,Afun);
  condl = snorm(M,@(x)rskelfr_mvl(F,x,'n'),@(x)rskelfr_mvl(F,x,'c')) ...
        * snorm(M,@(x)rskelfr_svl(F,x,'n'),@(x)rskelfr_svl(F,x,'c'));
  condu = snorm(N,@(x)rskelfr_mvu(F,x,'n'),@(x)rskelfr_mvu(F,x,'c')) ...
        * snorm(N,@(x)rskelfr_svu(F,x,'n'),@(x)rskelfr_svu(F,x,'c'));
  fprintf('uls: %10.4e / %10.4e / %10.4e / %10.4e\n', ...
          norm(B - C)/norm(B),norm(X),condl,condu)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
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