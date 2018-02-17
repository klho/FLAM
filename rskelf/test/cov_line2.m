% Matern covariance function on the unit line.

function cov_line2(n,occ,p,rank_or_tol,symm,noise,scale,spdiag)

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
  if nargin < 8 || isempty(spdiag)
    spdiag = 0;
  end

  % initialize
  x = (1:n)/n;
  N = size(x,2);
  proxy = linspace(1.5,2.5,p);
  proxy = [-proxy proxy];

  % factor matrix
  Afun = @(i,j)Afun2(i,j,x,noise);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun2(x,slf,nbr,l,ctr,proxy);
  opts = struct('symm',symm,'verb',1);
  F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

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
  rskelf_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(mv(x) - rskelf_mv(F,x)),[],[],1);
  e = e/snorm(N,mv,[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  rskelf_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - mv(rskelf_sv(F,x))),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic
    rskelf_cholmv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(rskelf_mv(F,x) ...
                           - rskelf_cholmv(F,rskelf_cholmv(F,x,'c'))),[],[],1);
    e = e/snorm(N,@(x)(rskelf_mv(F,x)),[],[],1);
    fprintf('cholmv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic
    rskelf_cholsv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(rskelf_sv(F,x) ...
                           - rskelf_cholsv(F,rskelf_cholsv(F,x),'c')),[],[],1);
    e = e/snorm(N,@(x)(rskelf_sv(F,x)),[],[],1);
    fprintf('cholsv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)
  end

  % compute log-determinant
  tic
  ld = rskelf_logdet(F);
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
    D = rskelf_spdiag(F);
    t1 = toc;
  else
    D = rskelf_diag(F,0,opts);
  end
  Y = rskelf_mv(F,X);
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
    D = rskelf_spdiag(F,1);
    t2 = toc;
  else
    D = rskelf_diag(F,1,opts);
  end
  Y = rskelf_sv(F,X);
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
    dr = scale*abs(bsxfun(@minus,x',y));
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
  Kpxy = Kfun(pxy,x(slf));
end

% FFT multiplication
function y = mv2(F,x)
  n = length(x);
  y = ifft(F.*fft(x,2*n-1));
  y = y(1:n);
end