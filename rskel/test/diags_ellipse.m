% Second-kind integral equation on an ellipse, Laplace double-layer.

function diags_ellipse(n,occ,p,rank_or_tol,ratio)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 16384;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
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

  % compress matrix
  opts = struct('verb',1);
  F = rskel(@Afun,x,x,occ,rank_or_tol,@pxyfun,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test matrix apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskel_mv(F,X,'n');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  A = Afun(r,1:N);
  Y = rskel_mv(F,X,'n');
  Z = A*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mv:  %10.4e / %10.4e (s)\n',e,t)

  % test matrix adjoint apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  rskel_mv(F,X,'c');
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  A = Afun(1:N,r);
  Y = rskel_mv(F,X,'c');
  Z = A'*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mva: %10.4e / %10.4e (s)\n',e,t)

  % extract diagonal
  tic
  D = rskel_diags(F);
  t = toc;
  E = zeros(N,1);
  for i = 1:N
    E(i) = Afun(i,i);
  end
  e = norm(D - E)/norm(E);
  fprintf('diags: %10.4e / %10.4e (s)\n',e,t)

  % kernel function
  function K = Kfun(x,y,lp,nu)
    if nargin < 4
      nu = [];
    end
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

  % matrix entries
  function A = Afun(i,j)
    A = Kfun(x(:,i),x(:,j),'d',nu(:,j));
    if any(j)
      A = bsxfun(@times,A,h(j));
    end
    [I,J] = ndgrid(i,j);
    idx = I == J;
    A(idx) = -0.5*(1 + 1/(2*pi)*h(J(idx)).*kappa(J(idx)));
  end

  % proxy function
  function [Kpxy,nbr] = pxyfun(rc,rx,cx,slf,nbr,l,ctr)
    pxy = bsxfun(@plus,proxy*l,ctr');
    if strcmpi(rc,'r')
      Kpxy = Kfun(rx(:,slf),pxy,'s')*(2*pi/N);
    elseif strcmpi(rc,'c')
      Kpxy = bsxfun(@times,Kfun(pxy,cx(:,slf),'d',nu(:,slf)),h(slf));
    end
    dx = x(1,nbr) - ctr(1);
    dy = x(2,nbr) - ctr(2);
    dist = sqrt(dx.^2 + dy.^2);
    nbr = nbr(dist/l < 1.5);
  end
end