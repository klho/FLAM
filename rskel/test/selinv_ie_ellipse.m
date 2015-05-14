% Second-kind integral equation on an ellipse, Laplace double-layer.

function selinv_ie_ellipse(n,occ,p,rank_or_tol,ratio)

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
  fprintf('mem: %6.2f (MB)\n',w.bytes/1e6)

  % factor extended sparsification
  tic
  A = rskel_xsp(F);
  t = toc;
  w = whos('A');
  fprintf('xsp: %10.4e (s) / %6.2f (MB)\n',t,w.bytes/1e6);
  tic
  [L,U] = lu(A);
  t = toc;
  w = whos('L');
  spmem = w.bytes;
  w = whos('U');
  spmem = (spmem + w.bytes)/1e6;
  fprintf('lu: %10.4e (s) / %6.2f (MB)\n',t,spmem)

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
  Y = rskel_mv(F,X,'n');
  Z = Afun(r,1:N)*X;
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
  Y = rskel_mv(F,X,'c');
  Z = Afun(1:N,r)'*X;
  e = norm(Z - Y(r,:))/norm(Z);
  fprintf('mva: %10.4e / %10.4e (s)\n',e,t)

  % test matrix inverse apply accuracy
  X = rand(N,1);
  X = X/norm(X);
  tic
  sv(X);
  t = toc;
  X = rand(N,16);
  X = X/norm(X);
  r = randperm(N);
  r = r(1:min(N,128));
  Y = sv(X);
  Z = Afun(r,1:N)*Y;
  e = norm(X(r,:) - Z)/norm(X(r,:));
  fprintf('sv: %10.4e / %10.4e (s)\n',e,t)

  % generate field due to exterior sources
  m = 16;
  theta = (1:m)*2*pi/m;
  src = 2*[ratio*cos(theta); sin(theta)];
  q = rand(m,1);
  B = Kfun(x,src,'s')*q;

  % solve for surface density
  X = sv(B);

  % evaluate field at interior targets
  trg = 0.5*[ratio*cos(theta); sin(theta)];
  Y = bsxfun(@times,Kfun(trg,x,'d',nu),h)*X;

  % compare against exact field
  Z = Kfun(trg,src,'s')*q;
  e = norm(Z - Y)/norm(Z);
  fprintf('pde: %10.4e\n',e)

  % prepare for selected inversion
  m = 16;
  r = randi(N,m,2);
  X = zeros(N,m);
  for i = 1:m
    X(r(i,2),i) = 1;
  end
  S = zeros(m,1);
  T = zeros(m,1);
  tic
  Li = inv(L);
  Uic = inv(U)';
  t = toc;
  nzLi = nnz(Li(:,1:N))/N;
  nzUi = nnz(Uic(:,1:N))/N;
  w = whos('Li');
  spmem = w.bytes;
  w = whos('Uic');
  spmem = (spmem + w.bytes)/1e6;
  fprintf('inv: %10.4e / %10.4e / %10.4e (s) / %6.2f (MB)\n',nzLi,nzUi,t,spmem)

  % selected inversion
  tic
  for i = 1:m
    S(i) = dot(Uic(:,r(i,1)),Li(:,r(i,2)));
  end
  t = toc/m;
  Y = sv(X);
  for i = 1:m
    T(i) = Y(r(i,1),i);
  end
  e = norm(S - T)/norm(T);
  fprintf('selinv: %10.4e / %10.4e (s)\n',e,t)

  % diagonal inversion
  tic
  for i = 1:m
    S(i) = dot(Uic(:,r(i,2)),Li(:,r(i,2)));
  end
  t = toc/m;
  Y = sv(X);
  for i = 1:m
    T(i) = Y(r(i,2),i);
  end
  e = norm(S - T)/norm(T);
  fprintf('diaginv: %10.4e / %10.4e (s)\n',e,t)

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
      dx = cx(1,nbr) - ctr(1);
      dy = cx(2,nbr) - ctr(2);
    elseif strcmpi(rc,'c')
      Kpxy = bsxfun(@times,Kfun(pxy,cx(:,slf),'d',nu(:,slf)),h(slf));
      dx = rx(1,nbr) - ctr(1);
      dy = rx(2,nbr) - ctr(2);
    end
    dist = sqrt(dx.^2 + dy.^2);
    nbr = nbr(dist/l < 1.5);
  end

  % sparse LU solve
  function Y = sv(X)
    X = [X; zeros(size(A,1)-N,size(X,2))];
    Y = U\(L\X);
    Y = Y(1:N,:);
  end
end