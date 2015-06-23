% Thirteen-point stencil on the unit cube, constant-coefficient Poisson,
% Dirichlet boundary conditions.

function fd_cube4x(n,occ,rank_or_tol,skip,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 32;
  end
  if nargin < 2 || isempty(occ)
    occ = 64;
  end
  if nargin < 3 || isempty(rank_or_tol)
    rank_or_tol = 1e-6;
  end
  if nargin < 4 || isempty(skip)
    skip = 2;
  end
  if nargin < 5 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2,x3] = ndgrid((1:n-1)/n);
  x = [x1(:) x2(:) x3(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2 x3

  % set up indices
  idx = zeros(n+3,n+3,n+3);
  idx(3:n+1,3:n+1,3:n+1) = reshape(1:N,n-1,n-1,n-1);
  mid = 3:n+1;
  lft1 = 2:n;
  lft2 = 1:n-1;
  rgt1 = 4:n+2;
  rgt2 = 5:n+3;

  % interactions with one-left node
  Il1 = idx(mid ,mid,mid);
  Jl1 = idx(lft1,mid,mid);
  Sl1 = -4/3/h^2*ones(size(Il1));

  % interactions with two-left node
  Il2 = idx(mid ,mid,mid);
  Jl2 = idx(lft2,mid,mid);
  Sl2 = 1/12/h^2*ones(size(Il2));

  % interactions with one-right node
  Ir1 = idx(mid ,mid,mid);
  Jr1 = idx(rgt1,mid,mid);
  Sr1 = -4/3/h^2*ones(size(Ir1));

  % interactions with two-right node
  Ir2 = idx(mid ,mid,mid);
  Jr2 = idx(rgt2,mid,mid);
  Sr2 = 1/12/h^2*ones(size(Ir2));

  % interactions with one-down node
  Id1 = idx(mid,mid ,mid);
  Jd1 = idx(mid,lft1,mid);
  Sd1 = -4/3/h^2*ones(size(Id1));

  % interactions with two-down node
  Id2 = idx(mid,mid ,mid);
  Jd2 = idx(mid,lft2,mid);
  Sd2 = 1/12/h^2*ones(size(Id2));

  % interactions with one-up node
  Iu1 = idx(mid,mid ,mid);
  Ju1 = idx(mid,rgt1,mid);
  Su1 = -4/3/h^2*ones(size(Iu1));

  % interactions with two-up node
  Iu2 = idx(mid,mid ,mid);
  Ju2 = idx(mid,rgt2,mid);
  Su2 = 1/12/h^2*ones(size(Iu2));

  % interactions with one-back node
  Ib1 = idx(mid,mid,mid );
  Jb1 = idx(mid,mid,lft1);
  Sb1 = -4/3/h^2*ones(size(Ib1));

  % interactions with two-back node
  Ib2 = idx(mid,mid,mid );
  Jb2 = idx(mid,mid,lft2);
  Sb2 = 1/12/h^2*ones(size(Ib2));

  % interactions with one-front node
  If1 = idx(mid,mid,mid );
  Jf1 = idx(mid,mid,rgt1);
  Sf1 = -4/3/h^2*ones(size(If1));

  % interactions with two-front node
  If2 = idx(mid,mid,mid );
  Jf2 = idx(mid,mid,rgt2);
  Sf2 = 1/12/h^2*ones(size(If2));

  % interactions with self
  Im = idx(mid,mid,mid);
  Jm = idx(mid,mid,mid);
  Sm = -(Sl1 + Sl2 + Sr1 + Sr2 + Sd1 + Sd2 + Su1 + Su2 + Sb1 + Sb2 + Sf1 + Sf2);

  % form sparse matrix
  I = [Il1(:); Il2(:); Ir1(:); Ir2(:); Id1(:); Id2(:); Iu1(:); Iu2(:); ...
       Ib1(:); Ib2(:); If1(:); If2(:); Im(:)];
  J = [Jl1(:); Jl2(:); Jr1(:); Jr2(:); Jd1(:); Jd2(:); Ju1(:); Ju2(:); ...
       Jb1(:); Jb2(:); Jf1(:); Jf2(:); Jm(:)];
  S = [Sl1(:); Sl2(:); Sr1(:); Sr2(:); Sd1(:); Sd2(:); Su1(:); Su2(:); ...
       Sb1(:); Sb2(:); Sf1(:); Sf2(:); Sm(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx Il1 Jl1 Sl1 Il2 Jl2 Sl2 Ir1 Jr1 Sr1 Ir2 Jr2 Sr2 ...
            Id1 Jd1 Sd1 Id2 Jd2 Sd2 Iu1 Ju1 Su1 Iu2 Ju2 Su2 ...
            Ib1 Jb1 Sb1 Ib2 Jb2 Sb2 If1 Jf1 Sf1 If2 Jf2 Sf2 Im Jm Sm I J S

  % factor matrix
  opts = struct('skip',skip,'symm',symm,'verb',1);
  F = hifde3x(A,x,occ,rank_or_tol,opts);
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
end