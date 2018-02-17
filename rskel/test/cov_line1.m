% Squared exponential covariance function on the unit line.

function cov_line1(n,occ,p,rank_or_tol,symm,noise,scale)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(p)
    p = 8;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 5 || isempty(symm)
    symm = 'p';
  end
  if nargin < 6 || isempty(noise)
    noise = 1e-2;
  end
  if nargin < 7 || isempty(scale)
    scale = 100;
  end

  % initialize
  x = (1:n)/n;
  N = size(x,2);
  proxy = linspace(1.5,2.5,p);
  proxy = [-proxy proxy];

  % compress matrix
  Afun = @(i,j)Afun2(i,j,x,noise);
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
  a = Afun(1:n,1);
  B = zeros(2*n-1,1);
  B(1:n) = a;
  B(n+1:end) = flipud(a(2:n));
  G = fft(B);
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
  sv(X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - mv(sv(x))),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % prepare for selected inversion
  m = 16;
  r = randi(N,m,2);
  X = zeros(N,m);
  for i = 1:m
    X(r(i,2),i) = 1;
  end
  ei = sparse([],[],[],size(A,1),1,1);
  ej = sparse([],[],[],size(A,1),1,1);
  S = zeros(m,1);
  T = zeros(m,1);
  I = speye(size(A));
  tic
  if dolu
    Li  =  FA.L\I;
    Uic = (FA.U\I)';
    nz = 0.5*(nnz(Li(:,1:N)) + nnz(Uic(:,1:N)))/N;
    w = whos('Li');
    spmem = w.bytes;
    w = whos('Uic');
    spmem = (spmem + w.bytes)/1e6;
  else
    Li = FA.L\I;
    Di = FA.D\I;
    Pi = FA.P\I;
    nz = nnz(Li(:,1:N))/N;
    w = whos('Li');
    spmem = w.bytes;
    w = whos('Di');
    spmem = spmem + w.bytes;
    w = whos('Pi');
    spmem = (spmem + w.bytes)/1e6;
  end
  t = toc;
  fprintf('inv: %10.4e (s) / %10.4e / %6.2f (MB)\n',t,nz,spmem)

  % selected inversion
  tic
  for i = 1:m
    if dolu
      S(i) = dot(Uic(:,r(i,1)),Li(:,r(i,2)));
    else
      ei(r(i,1)) = 1;
      ej(r(i,2)) = 1;
      S(i) = dot(Li*(Pi*ei),Di*(Li*(Pi*ej)));
      ei(r(i,1)) = 0;
      ej(r(i,2)) = 0;
    end
  end
  t = toc/m;
  Y = sv(X);
  for i = 1:m
    T(i) = Y(r(i,1),i);
  end
  e = norm(S - T)/norm(T);
  fprintf('selinv: %10.4e / %10.4e (s)\n',e,t)

  % diagonal inversion
  tic
  for i = 1:m
    if dolu
      S(i) = dot(Uic(:,r(i,2)),Li(:,r(i,2)));
    else
      ei(r(i,2)) = 1;
      ej(r(i,2)) = 1;
      S(i) = dot(Li*(Pi*ei),Di*(Li*(Pi*ej)));
      ei(r(i,2)) = 0;
      ej(r(i,2)) = 0;
    end
  end
  t = toc/m;
  Y = sv(X);
  for i = 1:m
    T(i) = Y(r(i,2),i);
  end
  e = norm(S - T)/norm(T);
  fprintf('diaginv: %10.4e / %10.4e (s)\n',e,t)

  % kernel function
  function K = Kfun(x,y)
    dr = scale*abs(bsxfun(@minus,x',y));
    K = exp(-0.5*dr.^2);
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
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = bsxfun(@plus,proxy*l,ctr');
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf));
  end
end

% FFT multiplication
function y = mv2(F,x)
  n = length(x);
  y = ifft(F.*fft(x,2*n-1));
  y = y(1:n);
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