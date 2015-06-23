% Nine-point stencil on the unit square, constant-coefficient Poisson, Dirichlet
% boundary conditions.

function fd_square4x(n,occ,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2] = ndgrid((1:n-1)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2

  % set up indices
  idx = zeros(n+3,n+3);
  idx(3:n+1,3:n+1) = reshape(1:N,n-1,n-1);
  mid = 3:n+1;
  lft1 = 2:n;
  lft2 = 1:n-1;
  rgt1 = 4:n+2;
  rgt2 = 5:n+3;

  % interactions with one-left node
  Il1 = idx(mid ,mid);
  Jl1 = idx(lft1,mid);
  Sl1 = -4/3/h^2*ones(size(Il1));

  % interactions with two-left node
  Il2 = idx(mid ,mid);
  Jl2 = idx(lft2,mid);
  Sl2 = 1/12/h^2*ones(size(Il2));

  % interactions with one-right node
  Ir1 = idx(mid ,mid);
  Jr1 = idx(rgt1,mid);
  Sr1 = -4/3/h^2*ones(size(Ir1));

  % interactions with two-right node
  Ir2 = idx(mid ,mid);
  Jr2 = idx(rgt2,mid);
  Sr2 = 1/12/h^2*ones(size(Ir2));

  % interactions with one-down node
  Id1 = idx(mid,mid );
  Jd1 = idx(mid,lft1);
  Sd1 = -4/3/h^2*ones(size(Id1));

  % interactions with two-down node
  Id2 = idx(mid,mid );
  Jd2 = idx(mid,lft2);
  Sd2 = 1/12/h^2*ones(size(Id2));

  % interactions with one-up node
  Iu1 = idx(mid,mid );
  Ju1 = idx(mid,rgt1);
  Su1 = -4/3/h^2*ones(size(Iu1));

  % interactions with two-up node
  Iu2 = idx(mid,mid );
  Ju2 = idx(mid,rgt2);
  Su2 = 1/12/h^2*ones(size(Iu2));

  % interactions with self
  Im = idx(mid,mid);
  Jm = idx(mid,mid);
  Sm = -(Sl1 + Sl2 + Sr1 + Sr2 + Sd1 + Sd2 + Su1 + Su2);

  % form sparse matrix
  I = [Il1(:); Il2(:); Ir1(:); Ir2(:); Id1(:); Id2(:); Iu1(:); Iu2(:); Im(:)];
  J = [Jl1(:); Jl2(:); Jr1(:); Jr2(:); Jd1(:); Jd2(:); Ju1(:); Ju2(:); Jm(:)];
  S = [Sl1(:); Sl2(:); Sr1(:); Sr2(:); Sd1(:); Sd2(:); Su1(:); Su2(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il1 Jl1 Sl1 Il2 Jl2 Sl2 Ir1 Jr1 Sr1 Ir2 Jr2 Sr2 ...
            Id1 Jd1 Sd1 Id2 Jd2 Sd2 Iu1 Ju1 Su1 Iu2 Ju2 Su2 Im Jm Sm I J S

  % factor matrix
  opts = struct('symm',symm,'verb',1);
  F = mfx(A,x,occ,opts);
  w = whos('F');
  fprintf([repmat('-',1,80) '\n'])
  fprintf('mem: %6.2f (MB)\n', w.bytes/1e6)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic
  mf_mv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(A*x - mf_mv(F,x)),[],[],1);
  e = e/snorm(N,@(x)(A*x),[],[],1);
  fprintf('mv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic
  Y = mf_sv(F,X);
  t = toc;
  [e,niter] = snorm(N,@(x)(x - A*mf_sv(F,x)),[],[],1);
  fprintf('sv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

  if strcmpi(symm,'p')
    % NORM(F - C*C')/NORM(F)
    tic
    mf_cholmv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(mf_mv(F,x) ...
                           - mf_cholmv(F,mf_cholmv(F,x,'c'))),[],[],1);
    e = e/snorm(N,@(x)(mf_mv(F,x)),[],[],1);
    fprintf('cholmv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)

    % NORM(INV(F) - INV(C')*INV(C))/NORM(INV(F))
    tic
    mf_cholsv(F,X);
    t = toc;
    [e,niter] = snorm(N,@(x)(mf_sv(F,x) ...
                           - mf_cholsv(F,mf_cholsv(F,x),'c')),[],[],1);
    e = e/snorm(N,@(x)(mf_sv(F,x)),[],[],1);
    fprintf('cholsv: %10.4e / %4d / %10.4e (s)\n',e,niter,t)
  end

  % run CG
  [~,~,~,iter] = pcg(@(x)(A*x),X,1e-12,128);

  % run PCG
  tic
  [Z,~,~,piter] = pcg(@(x)(A*x),X,1e-12,32,@(x)(mf_sv(F,x)));
  t = toc;
  e1 = norm(Z - Y)/norm(Z);
  e2 = norm(X - A*Z)/norm(X);
  fprintf('cg: %10.4e / %10.4e / %4d (%4d) / %10.4e (s)\n',e1,e2, ...
          piter,iter,t)

  % compute log-determinant
  tic
  ld = mf_logdet(F);
  t = toc;
  fprintf('logdet: %22.16e / %10.4e (s)\n',ld,t)
end