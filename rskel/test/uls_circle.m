% Underdetermined least squares on the unit circle, Laplace sources.

function uls_circle(m,n,occ,p,rank_or_tol,store,delta)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 8192;
  end
  if nargin < 2 || isempty(n)
    n = 16384;
  end
  if nargin < 3 || isempty(occ)
    occ = 128;
  end
  if nargin < 4 || isempty(p)
    p = 64;
  end
  if nargin < 5 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 6 || isempty(store)
    store = 'a';
  end
  if nargin < 7 || isempty(delta)
    delta = 1e-12;
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
  A = [tau*A; speye(N) sparse(N,size(A,2)-N)];
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
  nC = Ms - N;
  X = rand(N,1);
  B = ifmm_mv(G,X,@Afun);

  % test pseudoinverse apply accuracy
  tic
  [Y,cres,niter] = ls([B; zeros(nC-M,1)]);
  t = toc;
  e = norm(B - ifmm_mv(G,Y,@Afun))/norm(B);
  fprintf('ls: %10.4e / %10.4e / %2d / %10.4e (s) / %10.4e / %10.4e\n',e, ...
          cres,niter,t,norm(X),norm(Y))

  % run LSQR
  [~,~,~,iter] = lsqr(@mv,B,1e-9,128);

  % run LSQR with initial guess
  tic
  [Z,~,~,piter] = lsqr(@mv,B,1e-9,32,[],[],Y);
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(B - ifmm_mv(G,Z,@Afun))/norm(B);
  fprintf('lsqr: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2,piter,iter,t)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
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
      dx = cx(1,nbr) - ctr(1);
      dy = cx(2,nbr) - ctr(2);
    elseif strcmp(rc,'c')
      Kpxy = Kfun(pxy,cx(:,slf));
      dx = rx(1,nbr) - ctr(1);
      dy = rx(2,nbr) - ctr(2);
    end
    dist = sqrt(dx.^2 + dy.^2);
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
    [Y,cres,niter] = lsedc(@lsfun,A(nC+1:end,:),zeros(N,n),A(1:nC,:),X,tau);
    Y = Y(1:N);
  end

  % matrix multiply for LSQR
  function y = mv(x,trans)
    if strcmp(trans,'notransp')
      y = ifmm_mv(G,x,@Afun,'n');
    elseif strcmp(trans,'transp')
      y = ifmm_mv(G,x,@Afun,'c');
    end
  end
end