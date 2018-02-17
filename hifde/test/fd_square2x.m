% Five-point stencil on the unit square, variable-coefficient Poisson, Dirichlet
% boundary conditions.

function fd_square2x(n,occ,rank_or_tol,skip,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(rank_or_tol)
    rank_or_tol = 1e-9;
  end
  if nargin < 4 || isempty(skip)
    skip = 2;
  end
  if nargin < 5 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2] = ndgrid((1:n-1)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2

  % set up conductivity field
  a = zeros(n+1,n+1);
  A = rand(n-1,n-1);
  A = fft2(A,2*n-3,2*n-3);
  [X,Y] = ndgrid(0:n-2);
  C = normpdf(X,0,4).*normpdf(Y,0,4);
  B = zeros(2*n-3,2*n-3);
  B(1:n-1,1:n-1) = C;
  B(1:n-1,n:end) = C( :   ,2:n-1);
  B(n:end,1:n-1) = C(2:n-1, :   );
  B(n:end,n:end) = C(2:n-1,2:n-1);
  B(:,n:end) = flipdim(B(:,n:end),2);
  B(n:end,:) = flipdim(B(n:end,:),1);
  B = fft2(B);
  A = ifft2(A.*B);
  A = A(1:n-1,1:n-1);
  idx = A > median(A(:));
  A( idx) = 1e+2;
  A(~idx) = 1e-2;
  a(2:n,2:n) = A;
  clear X Y A B C

  % set up indices
  idx = zeros(n+1,n+1);
  idx(2:n,2:n) = reshape(1:N,n-1,n-1);
  mid = 2:n;
  lft = 1:n-1;
  rgt = 3:n+1;

  % interactions with left node
  Il = idx(mid,mid);
  Jl = idx(lft,mid);
  Sl = -0.5/h^2*(a(lft,mid) + a(mid,mid));

  % interactions with right node
  Ir = idx(mid,mid);
  Jr = idx(rgt,mid);
  Sr = -0.5/h^2*(a(rgt,mid) + a(mid,mid));

  % interactions with bottom node
  Id = idx(mid,mid);
  Jd = idx(mid,lft);
  Sd = -0.5/h^2*(a(mid,lft) + a(mid,mid));

  % interactions with top node
  Iu = idx(mid,mid);
  Ju = idx(mid,rgt);
  Su = -0.5/h^2*(a(mid,rgt) + a(mid,mid));

  % interactions with self
  Im = idx(mid,mid);
  Jm = idx(mid,mid);
  Sm = -(Sl + Sr + Sd + Su);

  % form sparse matrix
  I = [Il(:); Ir(:); Id(:); Iu(:); Im(:)];
  J = [Jl(:); Jr(:); Jd(:); Ju(:); Jm(:)];
  S = [Sl(:); Sr(:); Sd(:); Su(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il Jl Sl Ir Jr Sr Id Jd Sd Iu Ju Su Im Jm Sm I J S

  % factor matrix
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = hifde2x(A,x,occ,rank_or_tol,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  hifde_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - hifde_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = hifde_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*hifde_sv(F,x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic
    hifde_cholmv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(hifde_mv(F,x) ...
                           - hifde_cholmv(F,hifde_cholmv(F,x,'c'))),[],[],1);
    e = e/snorm(N,@(x)(hifde_mv(F,x)),[],[],1);
    fprintf('cholmv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic
    hifde_cholsv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(hifde_sv(F,x) ...
                           - hifde_cholsv(F,hifde_cholsv(F,x),'c')),[],[],1);
    e = e/snorm(N,@(x)(hifde_sv(F,x)),[],[],1);
    fprintf('cholsv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)
  end

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@(x)(hifde_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)

  % compute log-determinant
  tic
  ld = hifde_logdet(F);
  t = toc;
  fprintf('logdet: %22.16e / %10.4e (s)\n',ld,t)

  % Gaussian PDF
  function y = normpdf(x,mu,sigma)
    y = exp(-0.5*((x - mu)./sigma).^2)./(sqrt(2*pi).*sigma);
  end
end