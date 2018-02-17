% Second-kind integral equation on the unit square, Helmholtz single-layer.

function ie_square3x(n,k,occ,p,rank_or_tol,skip,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(k)
    k = 2*pi*4;
  end
  if nargin < 3 || isempty(occ)
    occ = 64;
  end
  if nargin < 4 || isempty(p)
    p = 64;
  end
  if nargin < 5 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 6 || isempty(skip)
    skip = 2;
  end
  if nargin < 7 || isempty(symm)
    symm = 's';
  end

  % initialize
  [x1,x2] = ndgrid((1:n)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];
  clear x1 x2

  % set up potential
  V = exp(-32*((x(1,:) - 0.5).^2 + (x(2,:) - 0.5).^2))';
  sqrtb = k*sqrt(V);

  % compute diagonal quadratures
  h = 1/n;
  intgrnd = @(x,y)(0.25i*besselh(0,1,k*sqrt(x.^2 + y.^2)));
  if isoctave()
    intgrl_r = 4*dblquad(@(x,y)(real(intgrnd(x,y))),0,h/2,0,h/2);
    intgrl_i = 4*dblquad(@(x,y)(imag(intgrnd(x,y))),0,h/2,0,h/2);
    intgrl = intgrl_r + intgrl_i*1i;
  else
    intgrl = 4*dblquad(intgrnd,0,h/2,0,h/2);
  end

  % factor matrix
  Afun = @(i,j)Afun2(i,j,sqrtb);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun2(x,slf,nbr,l,ctr,proxy,sqrtb,symm);
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = hifie2x(Afun,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % set up FFT multiplication
  a = reshape(Afun_ti(1:N,1),n,n);
  B = zeros(2*n-1,2*n-1);
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(:,n+1:end) = flipdim(B(:,n+1:end),2);
  B(n+1:end,:) = flipdim(B(n+1:end,:),1);
  G = fft2(B);
  mv = @(x)mv2(G,x,sqrtb);
  mva = @(x)(conj(mv(conj(x))));

  % test accuracy using randomized power method
  X = rand(N,1) + 1i*rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  hifie_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(mv (x) - hifie_mv(F,x,'n')), ...
                      @(x)(mva(x) - hifie_mv(F,x,'c')));
  e = e/snorm(N,mv,mva);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = hifie_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - mv (hifie_sv(F,x,'n'))), ...
                      @(x)(x - mva(hifie_sv(F,x,'c'))));
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % run unpreconditioned GMRES
  [~,~,~,iter] = gmres(mv,X,[],1e-12,1024);

  % run preconditioned GMRES
  tic
  [Z,~,~,piter] = gmres(mv,X,[],1e-12,32,@(x)(hifie_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - mv(Z))/norm(X);
  fprintf('gmres: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter(2),iter(2),t)

  % kernel function
  function K = Kfun(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    K = 0.25i*besselh(0,1,k*sqrt(dx.^2 + dy.^2));
  end

  % translation-invariant part of matrix
  function [A,idx] = Afun_ti(i,j)
    A = Kfun(x(:,i),x(:,j),k)/N;
    [I,J] = ndgrid(i,j);
    idx = I == J;
    A(idx) = intgrl;
  end
end

% matrix entries
function A = Afun2(i,j,sqrtb)
  [A,idx] = Afun_ti(i,j);
  A = bsxfun(@times,sqrtb(i),bsxfun(@times,A,sqrtb(j)'));
  A(idx) = A(idx) + 1;
end

% proxy function
function [Kpxy,nbr] = pxyfun2(x,slf,nbr,l,ctr,proxy,sqrtb,symm)
  pxy = bsxfun(@plus,proxy*l,ctr');
  N = size(x,2);
  Kpxy = Kfun(pxy,x(:,slf))/N;
  Kpxy = bsxfun(@times,Kpxy,sqrtb(slf)');
  if strcmpi(symm,'n')
    Kpxy = [Kpxy; conj(Kpxy)];
  end
  dx = x(1,nbr) - ctr(1);
  dy = x(2,nbr) - ctr(2);
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end

% FFT multiplication
function y = mv2(F,x,sqrtb)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(sqrtb.*x,n,n),2*n-1,2*n-1));
  y = sqrtb.*reshape(y(1:n,1:n),N,1);
  y = y + x;
end