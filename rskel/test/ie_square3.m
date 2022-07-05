% Second-kind integral equation on the unit square, Helmholtz single-layer.
%
% This example solves the Lippmann-Schwinger equation for Helmholtz scattering
% on the unit square, which can be ill-conditioned even though it is formally
% second-kind. The problem is discretized as in IE_SQUARE1. We assume that the
% potential/velocity field is nonnegative so that the matrix can be symmetrized;
% the resulting matrix is square, complex, symmetric, and Toeplitz. This demo
% follows the same outline as in IE_SQUARE1.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - K: wavenumber (default: K = 2*PI*4)
%   - OCC: tree occupancy parameter (default: OCC = 128)
%   - P: number of proxy points (default: P = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-6)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'S')
%   - DOITER: whether to run unpreconditioned GMRES (default: DOITER = 1)

function ie_square3(n,k,occ,p,rank_or_tol,Tmax,symm,doiter)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(k), k = 2*pi*4; end
  if nargin < 3 || isempty(occ), occ = 128; end
  if nargin < 4 || isempty(p), p = 64; end
  if nargin < 5 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 6 || isempty(Tmax), Tmax = 2; end
  if nargin < 7 || isempty(symm), symm = 's'; end
  if nargin < 8 || isempty(doiter), doiter = 1; end

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2;  % grid points
  N = size(x,2);
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points

  % set up potential/velocity field
  V = exp(-32*((x(1,:) - 0.5).^2 + (x(2,:) - 0.5).^2))';
  sqrtb = k*sqrt(V);  % assume nonnegative

  % compute diagonal quadratures
  h = 1/n;
  intgrnd = @(x,y)(0.25i*besselh(0,1,k*sqrt(x.^2 + y.^2)));
  if isoctave()  % no complex integration in Octave
    intgrl_r = 4*dblquad(@(x,y)(real(intgrnd(x,y))),0,h/2,0,h/2);
    intgrl_i = 4*dblquad(@(x,y)(imag(intgrnd(x,y))),0,h/2,0,h/2);
    intgrl = intgrl_r + intgrl_i*1i;
  else
    intgrl = 4*dblquad(intgrnd,0,h/2,0,h/2);
  end

  % compress matrix
  Afun = @(i,j)Afun_(i,j,x,k,intgrl,sqrtb);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,k, ...
                                            sqrtb);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskel time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = reshape(Afun_ti(1:N,1,x,k,intgrl),n,n);
  B = zeros(2*n-1,2*n-1);  % zero-pad
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(n+1:end,:) = flip(B(n+1:end,:),1);
  B(:,n+1:end) = flip(B(:,n+1:end),2);
  G = fft2(B);
  mv  = @(x)mv_(G,x,sqrtb);
  mva = @(x)conj(mv(conj(x)));

  % test accuracy using randomized power method
  X = rand(N,1) + 1i*rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskel_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv (x) - rskel_mv(F,x,'n')), ...
                @(x)(mva(x) - rskel_mv(F,x,'c')));
  err = err/snorm(N,mv,mva);
  fprintf('rskel_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % build extended sparsification
  tic; [A,p,q] = rskel_xsp(F); t = toc;
  w = whos('A'); mem = w.bytes/1e6;
  fprintf('rskel_xsp:\n')
  fprintf('  build time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem);

  % factor extended sparsification
  dolu = F.symm == 'n';  % LU or LDL?
  if ~dolu
    if isoctave(), warning('No LDL in Octave; using LU.');
    else,          warning('No complex sparse LDL in MATLAB; using LU.');
    end
    dolu = 1;
    A = A + tril(A,-1).';
  end
  FA = struct('p',p,'q',q,'lu',dolu);
  tic
  if dolu, [FA.L,FA.U,FA.P] = lu(A);
  else,    [FA.L,FA.D,FA.P] = ldl(A);
  end
  t = toc;
  w = whos('FA'); mem = w.bytes/1e6;
  fprintf('  factor time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)
  sv = @(x,trans)sv_(FA,x,trans);  % linear solve function

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; sv(X,'n'); t = toc;  % for timing
  err = snorm(N,@(x)(x - mv (sv(x,'n'))),@(x)(x - mva(sv(x,'c'))));
  fprintf('  solve err/time: %10.4e / %10.4e (s)\n',err,t)

  % run unpreconditioned GMRES
  B = mv(X);
  iter(1:2) = nan;
  if doiter, [~,~,~,iter] = gmres(mv,B,32,1e-12,32); end

  % run preconditioned GMRES
  tic; [Y,~,~,piter] = gmres(mv,B,32,1e-12,32,@(x)sv(x,'n')); t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - mv(Y))/norm(B);
  fprintf('gmres:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n',err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',(piter(1)+1)*piter(2), ...
          (iter(1)+1)*iter(2))
end

% kernel function
function K = Kfun(x,y,k)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  K = 0.25i*besselh(0,1,k*sqrt(dx.^2 + dy.^2));
end

% translation-invariant part of matrix, i.e., without potential
function [A,diagidx] = Afun_ti(i,j,x,k,intgrl)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j),k)/N;  % area-weighted point interaction
  [I,J] = ndgrid(i,j);
  diagidx = I == J;             % indices for diagonal
  A(diagidx) = intgrl;          % replace diagonal with precomputed values
end

% matrix entries
function A = Afun_(i,j,x,k,intgrl,sqrtb)
  [A,diagidx] = Afun_ti(i,j,x,k,intgrl);  % translation-invariant part
  if isempty(A), return; end
  % scale by potential/velocity field
  A = sqrtb(i).*A.*sqrtb(j)';
  A(diagidx) = A(diagidx) + 1;            % add identity to diagonal
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,k,sqrtb)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, multiplied by row/column potential/velocity field
  % and scaled to match the matrix scale
  N = size(rx,2);
  if rc == 'r'
    Kpxy = sqrtb(slf).*Kfun(rx(:,slf),pxy,k)/N;
    dr = cx(:,nbr) - ctr;
  else
    Kpxy = Kfun(pxy,cx(:,slf),k).*sqrtb(slf)'/N;
    dr = rx(:,nbr) - ctr;
  end
  % proxy points form ellipse of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipse
  nbr = nbr(sum((dr./l).^2) < 1.5^2);
end

% FFT multiplication
function y = mv_(F,x,sqrtb)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(sqrtb.*x,n,n),2*n-1,2*n-1));
  y = sqrtb.*reshape(y(1:n,1:n),N,1);
  y = y + x;
end

% sparse LU/LDL solve
function Y = sv_(F,X,trans)
  N = size(X,1);
  if trans == 'n', p = F.p; q = F.q;
  else,            p = F.q; q = F.p;
  end
  X = [X(p,:); zeros(size(F.L,1)-N,size(X,2))];
  if F.lu
    if trans == 'n', Y = F.U \(F.L \(F.P *X));
    else,            Y = F.P'*(F.L'\(F.U'\X));
    end
  else
    Y = F.P*(F.L'\(F.D\(F.L\(F.P'*X))));
  end
  Y = Y(1:N,:);
  Y(q,:) = Y;
end