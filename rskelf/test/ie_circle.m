% Second-kind integral equation on the unit circle, Laplace double-layer.
%
% This example solves the Dirichlet Laplace problem on the unit disk by using a
% double-layer potential, which yields a second-kind boundary integral equation.
% The integral is discretized with the trapezoidal rule; the resulting matrix is
% square, real, symmetric, and circulant.
%
% This demo does the following in order:
%
%   - factor the matrix
%   - check multiply/solve error/time
%   - check PDE solve error by applying to known solution
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points (default: N = 16384)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - P: number of proxy points (default: P = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-12)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'H')

function ie_circle(N,occ,p,rank_or_tol,Tmax,symm)

  % set default parameters
  if nargin < 1 || isempty(N), N = 16384; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(p), p = 64; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(symm), symm = 'h'; end

  % initialize
  theta = (1:N)*2*pi/N; x = [cos(theta); sin(theta)];  % discretization points
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % factor matrix
  Afun = @(i,j)Afun_(i,j,x);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskelf time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  G = fft(Afun(1:N,1));
  mv = @(x)mv_(G,x);

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskelf_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(mv(x) - rskelf_mv(F,x)),[],[],1);
  err = err/snorm(N,mv,[],[],1);
  fprintf('rskelf_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; rskelf_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - mv(rskelf_sv(F,x))),@(x)(x - rskelf_sv(F,mv(x),'c')));
  fprintf('rskelf_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  % generate field due to exterior sources (PDE reference solution)
  m = 16;                                                  % number of sources
  theta = (1:m)*2*pi/m; src = 2*[cos(theta); sin(theta)];  % source points
  q = rand(m,1);          % random charges for each source point
  B = Kfun(x,src,'s')*q;  % field evaluated at boundary

  % solve for boundary density
  X = rskelf_sv(F,B);

  % evaluate field from solved density at interior targets
  trg = 0.5*[cos(theta); sin(theta)];  % target points
  Y = Kfun(trg,x,'d')*(2*pi/N)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  err = norm(Z - Y)/norm(Z);
  fprintf('pde solve err: %10.4e\n',err)
end

% kernel function
function K = Kfun(x,y,lp)
  dx = x(1,:)' - y(1,:);
  dy = x(2,:)' - y(2,:);
  dr = sqrt(dx.^2 + dy.^2);
  if lp == 's'      % single-layer: G
    K = -1/(2*pi)*log(dr);
  elseif lp == 'd'  % double-layer: dG/dn
    rdotn = dx.*y(1,:) + dy.*y(2,:);
    K = 1/(2*pi)*rdotn./dr.^2;
  end
end

% matrix entries
function A = Afun_(i,j,x)
  N = size(x,2);
  A = Kfun(x(:,i),x(:,j),'d')*(2*pi/N);  % trapezoidal rule
  [I,J] = ndgrid(i,j);
  A(I == J) = -0.5*(1 + 1/N);  % limit = identity + curvature
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy)
  pxy = proxy.*l + ctr;  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  N = size(x,2);
  Kpxy = Kfun(pxy,x(:,slf),'s')*(2*pi/N);
  % proxy points form ellipse of scaled "radius" 1.5 around current box
  % keep among neighbors only those within ellipse
  nbr = nbr(sum(((x(:,nbr) - ctr)./l).^2) < 1.5^2);
end

% FFT multiplication
function y = mv_(F,x)
  y = ifft(F.*fft(x));
end