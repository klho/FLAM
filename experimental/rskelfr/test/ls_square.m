% Least squares on the unit square, regularized Laplace sources.

function ls_square(m,n,delta,occ,p,rank_or_tol,rdpiv,store)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 16384;
  end
  if nargin < 2 || isempty(n)
    n = 8192;
  end
  if nargin < 3 || isempty(delta)
    delta = 1e-3;
  end
  if nargin < 4 || isempty(occ)
    occ = 128;
  end
  if nargin < 5 || isempty(p)
    p = 64;
  end
  if nargin < 6 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 7 || isempty(rdpiv)
    rdpiv = 'l';
  end
  if nargin < 8 || isempty(store)
    store = 'a';
  end

  % initialize
  rx = rand(2,m);
  cx = rand(2,n);
  M = size(rx,2);
  N = size(cx,2);
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];

  % factor matrix using RSKELFR
  Afun = @(i,j)Afun2(i,j,rx,cx,delta);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy,delta);
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
  t = toc;
  [e,niter] = snorm(N,@(x)(ifmm_mv(G,x,Afun,'n') - rskelfr_mv(F,x,'n')), ...
                      @(x)(ifmm_mv(G,x,Afun,'c') - rskelfr_mv(F,x,'c')));
  e = e/snorm(N,@(x)(ifmm_mv(G,x,Afun,'n')),@(x)(ifmm_mv(G,x,Afun,'c')));
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % residual error: NORM(A - A*PINV(F)*A)/NORM(A)
  [e1,niter1] = snorm(N,
    @(x)(ifmm_mv(G,x,Afun,'n') - ...
         ifmm_mv(G,rskelfr_sv(F,ifmm_mv(G,x,Afun,'n'),'n'),Afun,'n')), ...
    @(x)(ifmm_mv(G,x,Afun,'c') - ...
         ifmm_mv(G,rskelfr_sv(F,ifmm_mv(G,x,Afun,'c'),'c'),Afun,'c')));
  e1 = e1/snorm(N,@(x)(ifmm_mv(G,x,Afun,'n')),@(x)(ifmm_mv(G,x,Afun,'c')));
  % solution error: NORM(I - PINV(F)*A)
  [e2,niter2] = snorm(N,
    @(x)(x - rskelfr_sv(F,ifmm_mv(G,x,Afun,'n'))), ...
    @(x)(x - ifmm_mv(G,rskelfr_sv(F,x,'c'),Afun,'c')));
  fprintf('ls: %10.4e / %4d / %10.4e / %4d\n',e1,niter1,e2,niter2)

  % concrete example
  B = ifmm_mv(G,X,Afun);
  tic
  Y = rskelfr_sv(F,B);
  t = toc;
  e1 = norm(B - ifmm_mv(G,Y,Afun))/norm(B);
  e2 = norm(X - Y);
  fprintf('ex: %10.4e / %10.4e / %10.4e / %10.4e (s)\n',e1,e2,norm(Y),t)

  if ~isoctave()
    % run LSQR
    mv = @(x,trans)mv_lsqr(G,x,trans,Afun);
    [~,~,~,iter] = lsqr(mv,B,1e-9,128);

    % run LSQR with initial guess
    tic
    [Z,~,~,piter] = lsqr(mv,B,1e-9,32,[],[],Y);
    t = toc;
    fprintf('lsqr')
  else
    warning('No LSQR in Octave.')

    % run CG
    [~,~,~,iter] = pcg(@(x)(ifmm_mv(G,ifmm_mv(G,x,Afun,'n'),Afun,'c')), ...
                       ifmm_mv(G,B,Afun,'c'),1e-9,128);

    % run preconditioned GMRES
    tic
    [Z,~,~,piter] = gmres(@(x)(rskelfr_sv(F,ifmm_mv(G,x,Afun))),Y,[],1e-9,32);
    t = toc;
    fprintf('cg/gmres')
  end
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(B - ifmm_mv(G,Z,Afun))/norm(B);
  fprintf(': %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2,piter(2),iter,t)
end

% kernel function
function K = Kfun(x,y,delta)
  dx = bsxfun(@minus,x(1,:)',y(1,:));
  dy = bsxfun(@minus,x(2,:)',y(2,:));
  K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2 + delta^2));
end

% matrix entries
function A = Afun2(i,j,rx,cx,delta)
  A = Kfun(rx(:,i),cx(:,j),delta);
end

% proxy function
function [Kpxy,nbr] = pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy,delta)
  pxy = bsxfun(@plus,proxy*l,ctr');
  if strcmpi(rc,'r')
    Kpxy = Kfun(rx(:,slf),pxy,delta);
    dx = cx(1,nbr) - ctr(1);
    dy = cx(2,nbr) - ctr(2);
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf),delta);
    dx = rx(1,nbr) - ctr(1);
    dy = rx(2,nbr) - ctr(2);
  end
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end

% matrix multiply for LSQR
function y = mv_lsqr(F,x,trans,Afun)
  if strcmpi(trans,'notransp')
    y = ifmm_mv(F,x,Afun,'n');
  elseif strcmpi(trans,'transp')
    y = ifmm_mv(F,x,Afun,'c');
  end
end