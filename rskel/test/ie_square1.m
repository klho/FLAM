% First-kind integral equation on the unit square, Laplace single-layer.

function ie_square1(n,occ,p,rank_or_tol,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
  end
  if nargin < 3 || isempty(p)
    p = 64;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 5 || isempty(symm)
    symm = 's';
  end

  % initialize
  [x1,x2] = ndgrid((1:n)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];
  clear x1 x2

  % compute diagonal quadratures
  h = 1/n;
  intgrl = 4*dblquad(@(x,y)(-1/(2*pi)*log(sqrt(x.^2 + y.^2))),0,h/2,0,h/2);

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x,intgrl);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('symm',symm,'verb',1);
  F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % factor extended sparsification
  tic
  A = rskel_xsp(F);
  t = toc;
  w = whos('A');
  fprintf('xsp: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6);
  dolu = strcmpi(F.symm,'n');
  if ~dolu && isoctave
    dolu = 1;
    A = A + tril(A,-1)';
  end
  FA = struct('lu',dolu);
  tic
  if dolu
    [FA.L,FA.U] = lu(A);
  else
    [FA.L,FA.D,FA.P] = ldl(A);
  end
  t = toc;
  if dolu
    w = whos('FA.L');
    spmem = w.bytes;
    w = whos('FA.U');
    spmem = (spmem + w.bytes)/1e6;
  else
    w = whos('FA.L');
    spmem = w.bytes;
    w = whos('FA.D');
    spmem = (spmem + w.bytes)/1e6;
  end
  fprintf('lu/ldl: %10.4e (s) / %6.2f (MB)\n',t,spmem)
  sv = @(x)sv2(FA,x);

  % set up FFT multiplication
  a = reshape(Afun(1:N,1),n,n);
  B = zeros(2*n-1,2*n-1);
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(:,n+1:end) = flipdim(B(:,n+1:end),2);
  B(n+1:end,:) = flipdim(B(n+1:end,:),1);
  G = fft2(B);
  mv = @(x)mv2(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  rskel_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(mv(x) - rskel_mv(F,x)),[],[],1);
  e = e/snorm(N,mv,[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = sv(X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - mv(sv(x))),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run unpreconditioned GMRES
  [~,~,~,iter] = gmres(mv,X,[],1e-12,128);

  % run preconditioned GMRES
  tic
  [Z,~,~,piter] = gmres(mv,X,[],1e-12,32,sv);
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - mv(Z))/norm(X);
  fprintf('gmres: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter(2),iter(2),t)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
  end
end

% matrix entries
function A = Afun2(i,j,x,intgrl)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j))/N;
  [I,J] = ndgrid(i,j);
  A(I == J) = intgrl;
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  N = size(rx,2);
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy)/N;
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf))/N;
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv2(F,x)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(x,n,n),2*n-1,2*n-1));
  y = reshape(y(1:n,1:n),N,1);
end

% sparse LU solve
function Y = sv2(F,X)
  N = size(X,1);
  X = [X; zeros(size(F.L,1)-N,size(X,2))];
  if F.lu
    Y = F.U\(F.L\X);
  else
    Y = F.P*(F.L'\(F.D\(F.L\(F.P'*X))));
  end
  Y = Y(1:N,:);
end