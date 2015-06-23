% Five-point stencil on the unit square, constant-coefficient Poisson,
% Dirichlet-Neumann boundary conditions.

function fd_square5x(n,occ,symm)

  % set default parameters
  if nargin < 1 || isempty(n)
    n = 128;
  end
  if nargin < 2 || isempty(occ)
    occ = 128;
  end
  if nargin < 3 || isempty(symm)
    symm = 'p';
  end

  % initialize
  [x1,x2] = ndgrid((0:n)/n,(1:n-1)/n);
  x = [x1(:) x2(:)]';
  N = size(x,2);
  h = 1/n;
  clear x1 x2

  % set up indices
  idx = zeros(n+1,n+1);
  idx(1:n+1,2:n) = reshape(1:N,n+1,n-1);
  mid = 2:n;
  lft = 1:n-1;
  rgt = 3:n+1;

  % interior interactions with left node
  Il = idx(mid,mid);
  Jl = idx(lft,mid);
  Sl = -1/h^2*ones(size(Il));

  % interior interactions with right node
  Ir = idx(mid,mid);
  Jr = idx(rgt,mid);
  Sr = -1/h^2*ones(size(Ir));

  % interior interactions with bottom node
  Id = idx(mid,mid);
  Jd = idx(mid,lft);
  Sd = -1/h^2*ones(size(Id));

  % interior interactions with top node
  Iu = idx(mid,mid);
  Ju = idx(mid,rgt);
  Su = -1/h^2*ones(size(Iu));

  % interior interactions with self
  Im = idx(mid,mid);
  Jm = idx(mid,mid);
  Sm = -(Sl + Sr + Sd + Su);

  % left Neumann boundary interactions with right node
  Ir1 = idx(1,mid);
  Jr1 = idx(2,mid);
  Sr1 = -1/h^2*ones(size(Ir1));

  % left Neumann boundary interactions with bottom node
  Id1 = idx(1,mid);
  Jd1 = idx(1,lft);
  Sd1 = -0.5/h^2*ones(size(Id1));

  % left Neumann boundary interactions with top node
  Iu1 = idx(1,mid);
  Ju1 = idx(1,rgt);
  Su1 = -0.5/h^2*ones(size(Iu1));

  % left Neumann boundary interactions with self
  Im1 = idx(1,mid);
  Jm1 = idx(1,mid);
  Sm1 = -(Sr1 + Sd1 + Su1);

  % right Neumann boundary interactions with left node
  Il2 = idx(n+1,mid);
  Jl2 = idx(n,mid);
  Sl2 = -1/h^2*ones(size(Il2));

  % right Neumann boundary interactions with bottom node
  Id2 = idx(n+1,mid);
  Jd2 = idx(n+1,lft);
  Sd2 = -0.5/h^2*ones(size(Id2));

  % right Neumann boundary interactions with top node
  Iu2 = idx(n+1,mid);
  Ju2 = idx(n+1,rgt);
  Su2 = -0.5/h^2*ones(size(Iu2));

  % right Neumann boundary interactions with self
  Im2 = idx(n+1,mid);
  Jm2 = idx(n+1,mid);
  Sm2 = -(Sl2 + Sd2 + Su2);

  % form sparse matrix
  I = [Il(:); Ir(:); Id(:); Iu(:); Im(:);
       Ir1(:); Id1(:); Iu1(:); Im1(:);
       Il2(:); Id2(:); Iu2(:); Im2(:)];
  J = [Jl(:); Jr(:); Jd(:); Ju(:); Jm(:);
       Jr1(:); Jd1(:); Ju1(:); Jm1(:);
       Jl2(:); Jd2(:); Ju2(:); Jm2(:)];
  S = [Sl(:); Sr(:); Sd(:); Su(:); Sm(:);
       Sr1(:); Sd1(:); Su1(:); Sm1(:);
       Sl2(:); Sd2(:); Su2(:); Sm2(:)];
  idx = find(J > 0);
  I = I(idx);
  J = J(idx);
  S = S(idx);
  A = sparse(I,J,S,N,N);
  clear idx ...
        Il Jl Sl Ir Jr Sr Id Jd Sd Iu Ju Su Im Jm Sm ...
        Ir1 Jr1 Sr1 Id1 Jd1 Sd1 Iu1 Ju1 Su1 Im1 Jm1 Sm1 ...
        Il2 Jl2 Sl2 Id2 Jd2 Sd2 Iu2 Ju2 Su2 Im2 Jm2 Sm2 ...
        I J S

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