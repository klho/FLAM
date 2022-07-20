% Overdetermined least squares on the unit line, thin-plate splines.
%
% This example solves a Tikhonov-regularized overdetermined least squares
% problem arising from thin-plate spline kernel fitting of values at equispaced
% points on the unit line by sources on an another equispaced grid. The
% resulting matrix is rectangular (tall-and-skinny) and real; we use an FMM as a
% fast computational reference.
%
% This demo does the following in order:
%
%   - compress the matrix
%   - check multiply error/time
%   - build/factor extended sparsification (regularized OLS form)
%   - check pseudoinverse solve error/time
%   - compare LSQR/CG with/without initial guess from approximate solve
%
% Inputs (defaults are used if not provided or set empty):
%
%   - M: number of row points (default: M = 16384)
%   - N: number of column points (default: N = 8192)
%   - LAMBDA: Tikhonov regularization (default: LAMBDA = 0.01)
%   - OCC: tree occupancy parameter (default: OCC = 128)
%   - P: half-number of proxy points (default: P = 8)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-12)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - STORE: FMM storage mode (default: STORE = 'A')
%   - DOITER: whether to run naive LSQR/CG (default: DOITER = 1)

function ols_line(M,N,lambda,occ,p,rank_or_tol,Tmax,store,doiter)

  % set default parameters
  if nargin < 1 || isempty(M), M = 16384; end
  if nargin < 2 || isempty(N), N =  8192; end
  if nargin < 3 || isempty(lambda), lambda = 0.01; end
  if nargin < 4 || isempty(occ), occ = 128; end
  if nargin < 5 || isempty(p), p = 8; end
  if nargin < 6 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 7 || isempty(Tmax), Tmax = 2; end
  if nargin < 8 || isempty(store), store = 'a'; end
  if nargin < 9 || isempty(doiter), doiter = 1; end

  % initialize
  rx = (1:M)/M;                                         % row points
  cx = (1:N)/N;                                         % col points
  proxy = linspace(1.5,2.5,p); proxy = [-proxy proxy];  % proxy points
  % reference proxy points are for unit box [-1, 1]

  % compress matrix using RSKEL
  Afun = @(i,j)Afun_(i,j,rx,cx);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('Tmax',Tmax,'verb',1);
  tic; F = rskel(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskel time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % compress matrix using IFMM
  rank_or_tol = max(rank_or_tol*1e-2,1e-15);  % higher accuracy for reference
  opts = struct('Tmax',Tmax,'store',store);
  tic; G = ifmm(Afun,rx,cx,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('G'); mem = w.bytes/1e6;
  fprintf('ifmm time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskel_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(ifmm_mv(G,x,Afun,'n') - rskel_mv(F,x,'n')), ...
                @(x)(ifmm_mv(G,x,Afun,'c') - rskel_mv(F,x,'c')));
  err = err/snorm(N,@(x)ifmm_mv(G,x,Afun,'n'),@(x)ifmm_mv(G,x,Afun,'c'));
  fprintf('rskel_mv err/time: %10.4e / %10.4e (s)\n',err,t)
  tic; ifmm_mv(G,X,Afun); t = toc;
  fprintf('ifmm_mv time: %10.4e (s)\n',t)

  % build extended sparsification
  tau = eps^(-1/3);
  tic
  [A,p,q] = rskel_xsp(F);
  A = [tau*A(M+1:end,:); A(1:M,:); lambda*speye(N) sparse(N,size(A,2)-N)];
  t = toc;
  w = whos('A'); mem = w.bytes/1e6;
  fprintf('rskel_xsp:\n')
  fprintf('  build time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem);

  % factor extended sparsification
  tic; R = qr(A,0); t = toc;
  w = whos('R'); mem = w.bytes/1e6;
  fprintf('  qr time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)
  FA = struct('A',A,'p',p,'q',q,'N',N,'tau',tau,'R',R);
  ls = @(X)ls_(FA,X);  % least squares solve function

  % test pseudoinverse apply accuracy
  B = ifmm_mv(G,X,Afun);                 % random right-hand side in range
  tic; [Y,cres,niter] = ls(B); t = toc;  % least squares solve
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - ifmm_mv(G,Y,Afun))/norm(B);
  fprintf('ls:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
  fprintf('  constraint resid/iter/soln norm: %10.4e / %d / %10.4e\n', ...
          norm(cres),niter,norm(Y))

  iter = nan;
  if ~isoctave()
    C = [B; zeros(N,1)];
    mv = @(x,trans)mv_lsqr(G,x,trans,Afun,M,lambda);

    % run LSQR
    if doiter, [~,~,~,iter] = lsqr(mv,C,1e-9,128); end

    % run LSQR with initial guess from pseudoinverse
    tic; [Z,~,~,piter] = lsqr(mv,C,1e-9,32,[],[],Y); t = toc;
    fprintf('lsqr:\n')
  else
    warning('No LSQR in Octave.')

    C = ifmm_mv(G,B,Afun,'c');
    mv = @(x)mv_cg(G,x,Afun,lambda);

    % run CG (on normal equations)
    if doiter, [~,~,~,iter] = pcg(mv,C,1e-9,128); end

    % run CG with initial guess from pseudoinverse
    tic; [Z,~,~,piter] = pcg(mv,C,1e-9,32,[],[],Y); t = toc;
    fprintf('cg:\n')
  end
  err1 = norm(X - Z)/norm(X);
  err2 = norm(B - ifmm_mv(G,Z,Afun))/norm(B);
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
  fprintf('  init/uninit iter: %d / %d\n',piter,iter)
end

% kernel function
function K = Kfun(x,y)
  dr = abs(x' - y);
  K = dr.^2.*log(dr);
  K(dr == 0) = 0;  % limit
end

% matrix entries
function A = Afun_(i,j,rx,cx)
  A = Kfun(rx(:,i),cx(:,j));
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  if rc == 'r'
    Kpxy = Kfun(rx(:,slf),pxy);
    dr = cx(:,nbr) - ctr;
  else
    Kpxy = Kfun(pxy,cx(:,slf));
    dr = rx(:,nbr) - ctr;
  end
  % proxy points form interval of scaled radius 1.5 around current box
  % keep among neighbors only those within interval
  nbr = nbr(abs(dr)/l < 1.5);
end

% weighted least squares solve
function x = lsfun(A,R,b)
  x = R\(R'\(A'*b));              % normal equation solve with one step of
  x = x + R\(R'\(A'*(b - A*x)));  % iterative refinement for accuracy
end

% equality-constrained least squares solve
function [Y,cres,niter] = ls_(F,X)
  [M,p] = size(X);
  X = [X(F.p,:); zeros(F.N,p)];  % for regularization
  nc = size(F.A,1) - M - F.N;    % number of constraints
  % deferred correction for iterated weighted least squares
  [Y,cres,niter] = lsedc(@(b)lsfun(F.A,F.R,b),F.A(nc+1:end,:),X, ...
                         F.A(1:nc,:)/F.tau,zeros(nc,p),F.tau);
  Y = Y(1:F.N,:);
  Y(F.q,:) = Y;
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