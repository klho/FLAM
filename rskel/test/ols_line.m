% Overdetermined least squares on the unit line, thin-plate splines with
% Tikhonov regularization.

function ols_line(m,n,lambda,occ,p,rank_or_tol,store)

  % set default parameters
  if nargin < 1 || isempty(m)
    m = 16384;
  end
  if nargin < 2 || isempty(n)
    n = 8192;
  end
  if nargin < 3 || isempty(lambda)
    lambda = 0.01;
  end
  if nargin < 4 || isempty(occ)
    occ = 128;
  end
  if nargin < 5 || isempty(p)
    p = 8;
  end
  if nargin < 6 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 7 || isempty(store)
    store = 'a';
  end

  % initialize
  rx = rand(1,m);
  cx = (1:n)/n;
  M = size(rx,2);
  N = size(cx,2);
  proxy = 1.5 + ((1:p) - 1)/p;
  proxy = [-proxy proxy];

  % compress matrix using RSKEL
  Afun = @(i,j)Afun2(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun2(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('verb',1);
  F = rskel(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts);
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
  rskel_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(ifmm_mv(G,x,Afun,'n') - rskel_mv(F,x,'n')), ...
                      @(x)(ifmm_mv(G,x,Afun,'c') - rskel_mv(F,x,'c')));
  e = e/snorm(N,@(x)(ifmm_mv(G,x,Afun,'n')),@(x)(ifmm_mv(G,x,Afun,'c')));
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

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
  nc = size(A,1) - M - N;
  ls = @(X)ls2(A,R,X,N,nc,tau);

  % test pseudoinverse apply accuracy
  B = ifmm_mv(G,X,Afun);
  tic
  [Y,cres,niter] = ls(B);
  t = toc;
  e1 = norm(B - ifmm_mv(G,Y,Afun))/norm(B);
  e2 = norm(X - Y);
  fprintf('ls: %10.4e / %10.4e / %2d / %10.4e / %10.4e / %10.4e (s)\n', ...
          e1,cres,niter,e2,norm(Y),t)

  if ~isoctave()
    % run LSQR
    C = [X; zeros(N,1)];
    mv = @(x,trans)mv_lsqr(G,x,trans,Afun,M,lambda);
    [~,~,~,iter] = lsqr(mv,C,1e-9,128);

    % run LSQR with initial guess
    tic
    [Z,~,~,piter] = lsqr(mv,C,1e-9,32,[],[],Y);
    t = toc;
    fprintf('lsqr')
  else
    warning('No LSQR in Octave.')

    % run CG
    C = ifmm_mv(G,B,Afun,'c');
    mv = @(x)mv_cg(G,x,Afun,lambda);
    [~,~,~,iter] = pcg(mv,C,1e-9,128);

    % run CG with initial guess
    tic
    [Z,~,~,piter] = pcg(mv,C,1e-9,32,[],[],Y);
    t = toc;
    fprintf('cg')
  end
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(B - ifmm_mv(G,Z,Afun))/norm(B);
  fprintf(': %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2,piter,iter,t)
end

% kernel function
function K = Kfun(x,y)
  dr = abs(bsxfun(@minus,x',y));
  K = dr.^2.*log(dr);
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
  elseif strcmpi(rc,'c')
    Kpxy = Kfun(pxy,cx(:,slf));
  end
end

% least squares solve
function x = lsfun(A,R,b)
  x = R\(R'\(A'*b));
  x = x + R\(R'\(A'*(b - A*x)));
end

% equality-constrained least squares solve
function [Y,cres,niter] = ls2(A,R,X,N,nc,tau)
  n = size(X,2);
  X = [X; zeros(N,n)];
  [Y,cres,niter] = lsedc(@(b)lsfun(A,R,b),A(nc+1:end,:),X,A(1:nc,:), ...
                         zeros(nc,n),tau);
  Y = Y(1:N,:);
end

% matrix multiply for LSQR
function y = mv_lsqr(F,x,trans,Afun,M,lambda)
  if strcmpi(trans,'notransp')
    y = [ifmm_mv(F,x,Afun,'n'); lambda*x];
  elseif strcmpi(trans,'transp')
    y = ifmm_mv(F,x(1:M),Afun,'c') + lambda*x(M+1:end);
  end
end

% matrix multiply for CG
function y = mv_cg(F,x,Afun,lambda)
  y = ifmm_mv(F,ifmm_mv(F,x,Afun,'n'),Afun,'c') + lambda^2*x;
end