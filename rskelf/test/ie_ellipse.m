% Second-kind integral equation on an ellipse, Laplace double-layer.
%
% This is a slight generalization of IE_CIRCLE, where now the problem geometry
% is an ellipse with user-specified aspect ratio. Note that the matrix is no
% longer Toeplitz.

function ie_ellipse(n,occ,p,rank_or_tol,symm,ratio)

  % set default parameters
  if nargin < 1 || isempty(n), n = 16384; end  % number of points
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(p), p = 64; end  % number of proxy points
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 5 || isempty(symm), symm = 'h'; end  % symmetric/Hermitian
  if nargin < 6 || isempty(ratio), ratio = 2; end

  % initialize
  theta = (1:n)*2*pi/n;
  x = [ratio*cos(theta); sin(theta)];  % discretization points
  N = size(x,2);
  % unit normal
  nu = [cos(theta); ratio*sin(theta)];
  h = sqrt(nu(1,:).^2 + nu(2,:).^2);
  nu = bsxfun(@rdivide,nu,h);
  kappa = ratio./h.^3;  % curvature
  h = 2*pi/n*h;         % arc length
  theta = (1:p)*2*pi/p; proxy = 1.5*[cos(theta); sin(theta)];  % proxy points
  % reference proxy points are for unit box [-1, 1]^2

  % factor matrix
  Afun = @(i,j)Afun_(i,j,x,nu,h,kappa);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,proxy,nu,h);
  opts = struct('symm',symm,'verb',1);
  tic; F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskelf time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up accuracy tests
  X1 = rand(N,1); X1 = X1/norm(X1);  % for timing
  X = rand(N,16); X = X/norm(X);  % test against 16 vectors for robustness
  r = randperm(N); r = r(1:min(N,128));  % check up to 128 rows in result

  % test matrix apply accuracy
  tic; rskelf_mv(F,X1,'n'); t = toc;  % for timing
  Y = rskelf_mv(F,X,'n');
  Z = Afun(r,1:N)*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('rskelf_mv:\n')
  fprintf('  multiply err/time: %10.4e / %10.4e (s)\n',err,t)

  % test matrix adjoint apply accuracy
  tic; rskelf_mv(F,X1,'c'); t = toc;  % for timing
  Y = rskelf_mv(F,X,'c');
  Z = Afun(1:N,r)'*X;
  err = norm(Z - Y(r,:))/norm(Z);
  fprintf('  adjoint multiply err/time: %10.4e / %10.4e (s)\n',err,t)

  % test matrix inverse apply accuracy
  tic; rskelf_sv(F,X1,'n'); t = toc;  % for timing
  Y = rskelf_sv(F,X,'n');
  Z = Afun(r,1:N)*Y;
  err = norm(X(r,:) - Z)/norm(X(r,:));
  fprintf('rskelf_sv:\n')
  fprintf('  solve err/time: %10.4e / %10.4e (s)\n',err,t)

  % test matrix inverse adjoint apply accuracy
  tic; rskelf_sv(F,X1,'c'); t = toc;  % for timing
  Y = rskelf_sv(F,X,'c');
  Z = Afun(1:N,r)'*Y;
  err = norm(X(r,:) - Z)/norm(X(r,:));
  fprintf('  adjoint solve err/time: %10.4e / %10.4e (s)\n',err,t)

  % generate field due to exterior sources (PDE reference solution)
  m = 16;                 % number of sources
  theta = (1:m)*2*pi/m; src = 2*[ratio*cos(theta); sin(theta)];  % source points
  q = rand(m,1);          % random charges for each source point
  B = Kfun(x,src,'s')*q;  % field evaluated at boundary

  % solve for boundary density
  X = rskelf_sv(F,B);

  % evaluate field from solved density at interior targets
  trg = 0.5*[ratio*cos(theta); sin(theta)];  % target points
  Y = bsxfun(@times,Kfun(trg,x,'d',nu),h)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  err = norm(Z - Y)/norm(Z);
  fprintf('pde solve err: %10.4e\n',err)
end

% kernel function
function K = Kfun(x,y,lp,nu)
  dx = bsxfun(@minus,x(1,:)',y(1,:));
  dy = bsxfun(@minus,x(2,:)',y(2,:));
  dr = sqrt(dx.^2 + dy.^2);
  if strcmpi(lp,'s')      % single-layer: G
    K = -1/(2*pi)*log(dr);
  elseif strcmpi(lp,'d')  % double-layer: dG/dn
    rdotn = bsxfun(@times,dx,nu(1,:)) + bsxfun(@times,dy,nu(2,:));
    K = 1/(2*pi).*rdotn./dr.^2;
  end
end

% matrix entries
function A = Afun_(i,j,x,nu,h,kappa)
  A = Kfun(x(:,i),x(:,j),'d',nu(:,j));
  if ~isempty(j), A = bsxfun(@times,A,h(j)); end  % trapezoidal rule
  % limit = identity + curvature
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = -0.5*(1 + 1/(2*pi)*h(J(idx)).*kappa(J(idx)));
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,proxy,nu,h)
  pxy = bsxfun(@plus,proxy*l,ctr');  % scale and translate reference points
  % proxy interaction is kernel evaluation between proxy points and row/column
  % points being compressed, scaled to match the matrix scale
  Kpxy = bsxfun(@times,Kfun(pxy,x(:,slf),'d',nu(:,slf)),h(slf));
  dx = x(1,nbr) - ctr(1);
  dy = x(2,nbr) - ctr(2);
  % proxy points form circle of scaled radius 1.5 around current box
  % keep among neighbors only those within circle
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end