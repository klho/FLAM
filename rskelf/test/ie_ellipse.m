% Second-kind integral equation on an ellipse, Laplace double-layer.

function ie_ellipse(n,occ,p,rank_or_tol,ratio)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(p)
    p = 64;
  end
  if nargin < 4 || isempty(rank_or_tol)
    rank_or_tol = 1e-12;
  end
  if nargin < 5 || isempty(ratio)
    ratio = 2;
  end

  % initialize
  theta = (1:n)*2*pi/n;
  x = [ratio*cos(theta); sin(theta)];
  N = size(x,2);
  nu = [cos(theta); ratio*sin(theta)];
  h = sqrt(nu(1,:).^2 + nu(2,:).^2);
  nu = bsxfun(@rdivide,nu,h);
  kappa = ratio./h.^3;
  h = 2*pi/n*h;
  theta = (1:p)*2*pi/p;
  proxy = 1.5*[cos(theta); sin(theta)];

  % factor matrix
  Afun = @(i,j)Afun2(i,j,x,nu,h,kappa);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun2(x,slf,nbr,l,ctr,proxy,nu,h);
  opts = struct('verb',1);
  F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % test matrix apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskelf_mv(F,X,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  Y = rskelf_mv(F,X,'n');
  Z = Afun(r,1:N)*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix adjoint apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskelf_mv(F,X,'c');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  Y = rskelf_mv(F,X,'c');
  Z = Afun(1:N,r)'*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mva: %10.4e / %10.4e (s)\n',e,t)

  % test matrix inverse apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskelf_sv(F,X,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  Y = rskelf_sv(F,X,'n');
  Z = Afun(r,1:N)*Y;
  e = norm(X(r,:) - Z)/norm(X(r,:));
  fprintf('sv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix inverse adjoint apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskelf_sv(F,X,'c');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  Y = rskelf_sv(F,X,'c');
  Z = Afun(1:N,r)'*Y;
  e = norm(X(r,:) - Z)/norm(X(r,:));
  fprintf('sva: %10.4e / %10.4e (s)\n',e,t)

  % generate field due to exterior sources
  m = 16;
  theta = (1:m)*2*pi/m;
  src = 2*[ratio*cos(theta); sin(theta)];
  q = rand(m,1);
  B = Kfun(x,src,'s')*q;

  % solve for surface density
  X = rskelf_sv(F,B);

  % evaluate field at interior targets
  trg = 0.5*[ratio*cos(theta); sin(theta)];
  Y = bsxfun(@times,Kfun(trg,x,'d',nu),h)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  e = norm(Z - Y)/norm(Z);
  fprintf('pde: %10.4e\n',e)

  % kernel function
  function K = Kfun(x,y,lp,nu)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2 + dy.^2);
    if strcmpi(lp,'s')
      K = -1/(2*pi)*log(dr);
    elseif strcmpi(lp,'d')
      rdotn = bsxfun(@times,dx,nu(1,:)) + bsxfun(@times,dy,nu(2,:));
      K = 1/(2*pi).*rdotn./dr.^2;
    end
  end
end

% matrix entries
function A = Afun2(i,j,x,nu,h,kappa)
  A = Kfun(x(:,i),x(:,j),'d',nu(:,j));
  if any(j)
    A = bsxfun(@times,A,h(j));
  end
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = -0.5*(1 + 1/(2*pi)*h(J(idx)).*kappa(J(idx)));
end

% proxy function
function [Kpxy,nbr] = pxyfun2(x,slf,nbr,l,ctr,proxy,nu,h)
  pxy = bsxfun(@plus,proxy*l,ctr');
  Kpxy = bsxfun(@times,Kfun(pxy,x(:,slf),'d',nu(:,slf)),h(slf));
  dx = x(1,nbr) - ctr(1);
  dy = x(2,nbr) - ctr(2);
  dist = sqrt(dx.^2 + dy.^2);
  nbr = nbr(dist/l < 1.5);
end