% Second-kind integral equation on the unit square, Laplace single-layer.
%
% This is basically the same as IE_SQUARE1 but with the identity added to the
% system matrix in order to create an artificial second-kind volume integral
% operator. It skips the preconditioned GMRES comparison since the problem is
% well-conditioned.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 128)
%   - P: number of proxy points (default: P = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-6)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'S')

function ie_square2(n,occ,p,rank_or_tol,Tmax,symm)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 128; end
  if nargin < 3 || isempty(p), p = 64; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-6; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(symm), symm = 's'; end

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2;  % grid points
  N = size(x,2);
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % compute diagonal quadratures
  h = 1/n;
  intgrl = 4*dblquad(@(x,y)(-1/(2*pi)*log(sqrt(x.^2 + y.^2))),0,h/2,0,h/2);

  % compress matrix
  Afun = @(i,j)Afun_(i,j,x,intgrl);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskel time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = reshape(Afun(1:N,1),n,n);
  B = zeros(2*n-1,2*n-1);  % zero-pad
  B(  1:n  ,  1:n  ) = a;
  B(  1:n  ,n+1:end) = a( : ,2:n);
  B(n+1:end,  1:n  ) = a(2:n, : );
  B(n+1:end,n+1:end) = a(2:n,2:n);
  B(n+1:end,:) = flip(B(n+1:end,:),1);
  B(:,n+1:end) = flip(B(:,n+1:end),2);
  G = fft2(B);
  mv = @(x)mv_(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskel_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - rskel_mv(F,x)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('rskel_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % build extended sparsification
  tic; [A,p,q] = rskel_xsp(F); t = toc;
  w = whos('A'); mem = w.bytes/1e6;
  fprintf('rskel_xsp:\n')
  fprintf('  build time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem);

  % factor extended sparsification
  dolu = F.symm == 'n';  % LU or LDL?
  if ~dolu && isoctave()
    warning('No LDL in Octave; using LU.')
    dolu = 1;
    A = A + tril(A,-1)';
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
  err = snorm(N,@(x)(x - mv(sv(x,'n'))),@(x)(x - sv(mv(x),'c')));
  fprintf('  solve err/time: %10.4e / %10.4e (s)\n',err,t)
end

% kernel function
function K = Kfun(x,y)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  K = -1/(2*pi)*log(sqrt(dx.^2 + dy.^2));
end

% matrix entries
function A = Afun_(i,j,x,intgrl)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j))/N;  % area-weighted point interaction
  [I,J] = ndgrid(i,j);
  A(I == J) = 1 + intgrl;  % replace diagonal with identity + precomputed values
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  N = size(rx,2);
  if rc == 'r'
    Kpxy = Kfun(rx(:,slf),pxy)/N;
    dr = cx(:,nbr) - ctr;
  else
    Kpxy = Kfun(pxy,cx(:,slf))/N;
    dr = rx(:,nbr) - ctr;
  end
  % proxy points form ellipse of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipse
  nbr = nbr(sum((dr./l).^2) < 1.5^2);
end

% FFT multiplication
function y = mv_(F,x)
  N = length(x);
  n = sqrt(N);
  y = ifft2(F.*fft2(reshape(x,n,n),2*n-1,2*n-1));
  y = reshape(y(1:n,1:n),N,1);
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