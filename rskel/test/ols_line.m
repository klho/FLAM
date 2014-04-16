% Overdetermined least squares on the unit line, thin-plate splines with
% Tikhonov regularization.

function ols_line(m,n,occ,p,rank_or_tol,store,lambda)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 16384;
  end
  if nargin < 2 || isempty(n)
    n = 8192;
  end
  if nargin < 3 || isempty(occ)
    occ = 128;
  end
  if nargin < 4 || isempty(p)
    p = 8;
  end
  if nargin < 5 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 6 || isempty(store)
    store = 'a';
  end
  if nargin < 7 || isempty(lambda)
    lambda = 0.01;
  end

  % initialize
  rx = rand(1,m);
  cx = (1:n)/n;
  M = size(rx,2);
  N = size(cx,2);
  proxy = 1.5 + ((1:p) - 1)/p;
  proxy = [-proxy proxy];

  % compress matrix using RSKEL
  opts = struct('verb',1);
  F = rskel(@Afun,rx,cx,occ,rank_or_tol,@pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % compress matrix using IFMM
  opts = struct('store',store,'verb',1);
  G = ifmm(@Afun,rx,cx,2*occ,1e-15,@pxyfun_ifmm,opts);
  w = whos('G');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  rskel_mv(F,X);
  t1 = toc;
  tic
  ifmm_mv(G,X,@Afun);
  t2 = toc;
  [e,niter] = snorm(N,@(x)(ifmm_mv(G,x,@Afun,'n') - rskel_mv(F,x,'n')), ...
                      @(x)(ifmm_mv(G,x,@Afun,'c') - rskel_mv(F,x,'c')));
  e = e/snorm(N,@(x)(ifmm_mv(G,x,@Afun,'n')),@(x)(ifmm_mv(G,x,@Afun,'c')));
  fprintf('mv: %10.4e / %4d / %10.4e (s) / %10.4e (s)\n',e,niter,t1,t2)

  % factor extended sparsification
  tau = eps^(-1/3);
  tic
  A = rskel_xsp(F);
  A = [tau*A(M+1:end,:); A(1:M,:); lambda*speye(N) sparse(N,size(A,2)-N)];
  t = toc;
  w = whos('A');
  fprintf('xsp: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6);
  tic
  R = qr(A,0);
  t = toc;
  w = whos('R');
  fprintf('qr: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6)

  % set up right-hand side
  [Ms,Ns] = size(A);
  nC = Ms - M - N;
  X = ifmm_mv(G,rand(N,1),@Afun);
  X = X + 1e-2*randn(size(X));

  % test pseudoinverse apply accuracy
  tic
  [Y,cres,niter] = ls(X);
  t = toc;
  e = norm(X - ifmm_mv(G,Y,@Afun))/norm(X);
  fprintf('ls: %10.4e / %10.4e / %2d / %10.4e (s)\n',e,cres,niter,t)

  % run LSQR
  [~,~,~,iter] = lsqr(@mv,[X; zeros(N,1)],1e-9,128);

  % run LSQR with initial guess
  tic
  [Z,~,~,piter] = lsqr(@mv,[X; zeros(N,1)],1e-9,32,[],[],Y);
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - ifmm_mv(G,Z,@Afun))/norm(X);
  fprintf('lsqr: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2,piter,iter,t)

  % kernel function
  function K = Kfun(x,y)
    dr = abs(bsxfun(@minus,x',y));
    K = dr.^2.*log(dr);
  end

  % matrix entries
  function A = Afun(i,j)
    A = Kfun(rx(:,i),cx(:,j));
  end

  % proxy function
  function [Kpxy,nbr] = pxyfun(rc,rx,cx,slf,nbr,l,ctr)
    pxy = bsxfun(@plus,proxy*l,ctr');
    if strcmp(rc,'r')
      Kpxy = Kfun(rx(:,slf),pxy);
      dist = cx(:,nbr) - ctr;
    elseif strcmp(rc,'c')
      Kpxy = Kfun(pxy,cx(:,slf));
      dist = rx(:,nbr) - ctr;
    end
    nbr = nbr(dist/l < 1.5);
  end

  % proxy function for IFMM
  function K = pxyfun_ifmm(rc,rx,cx,slf,nbr,l,ctr)
    pxy = bsxfun(@plus,proxy*l,ctr');
    if strcmp(rc,'r')
      K = Kfun(rx(:,slf),pxy);
    elseif strcmp(rc,'c')
      K = Kfun(pxy,cx(:,slf));
    end
  end

  % least squares solve
  function x = lsfun(b)
    x = R\(R'\(A'*b));
    x = x + R\(R'\(A'*(b - A*x)));
  end

  % equality-constrained least squares solve
  function [Y,cres,niter] = ls(X)
    n = size(X,2);
    X = [X; zeros(N,n)];
    [Y,cres,niter] = lsedc(@lsfun,A(nC+1:end,:),X,A(1:nC,:),zeros(nC,n),tau);
    Y = Y(1:N);
  end

  % matrix multiply for LSQR
  function y = mv(x,trans)
    if strcmp(trans,'notransp')
      y = [ifmm_mv(G,x,@Afun,'n'); lambda*x];
    elseif strcmp(trans,'transp')
      y = ifmm_mv(G,x(1:M),@Afun,'c') + lambda*x(M+1:end);
    end
  end
end