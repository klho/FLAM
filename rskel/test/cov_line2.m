% Covariance matrix on the unit line, Matern 3/2 kernel.
%
% This is basically the same as COV_LINE1 but using the Matern 3/2 kernel.
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of points (default: N = 16384)
%   - OCC: tree occupancy parameter (default: OCC = 128)
%   - P: half-number of proxy points (default: P = 8)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-12)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - NOISE: nugget effect (default: NOISE = 1e-2)
%   - SCALE: kernel length scale (default: SCALE = 100)

function cov_line2(N,occ,p,rank_or_tol,Tmax,symm,noise,scale)

  % set default parameters
  if nargin < 1 || isempty(N), N = 16384; end
  if nargin < 2 || isempty(occ), occ = 128; end
  if nargin < 3 || isempty(p), p = 8; end
  if nargin < 4 || isempty(rank_or_tol), rank_or_tol = 1e-12; end
  if nargin < 5 || isempty(Tmax), Tmax = 2; end
  if nargin < 6 || isempty(symm), symm = 'p'; end
  if nargin < 7 || isempty(noise), noise = 1e-2; end
  if nargin < 8 || isempty(scale), scale = 100; end

  % initialize
  x = (1:N)/N;                                      % grid points
  R = 3/scale;                                      % annular width
  proxy = linspace(0,R,p); proxy = [-proxy proxy];  % proxy points
  % reference proxy points are for a single point at the origin only
  shift = 1.5*sign(proxy);  % reference shift for the unit box [-1, 1]^2

  % compress matrix
  Afun = @(i,j)Afun_(i,j,x,noise,scale);
  pxyfun = @(rc,rx,cx,slf,nbr,l,ctr)pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy, ...
                                            shift,scale);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskel(Afun,x,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskel time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % set up reference FFT multiplication
  a = Afun(1:N,1);
  B = zeros(2*N-1,1);  % zero-pad
  B(1:N) = a;
  B(N+1:end) = flip(a(2:N));
  G = fft(B);
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

  % prepare for selected/diagonal inversion
  % note: reported error is with respect to compressed solve -- should ideally
  %       be close O(eps), independently of compression tolerance
  m = 128;           % number of entries
  r = randi(N,m,2);  % random row/col index for each entry
  % reference comparison from compressed solve against coordinate vectors
  X = zeros(N,m);
  for i = 1:m, X(r(i,2),i) = 1; end
  Y = sv(X,'n');  % subselect r(:,1) or r(:,2) rows for selected/diagonal inv
  % coordinate vector storage in sparse form
  ei = sparse([],[],[],size(A,1),1,1);
  ej = sparse([],[],[],size(A,1),1,1);
  % solution storage
  S = zeros(m,1);
  T = zeros(m,1);

  % selected inversion
  % algorithm: sparse dot product of (inverse) matrix factor row/cols
  % complexity: O(LOG(N)) for each entry
  % note: error can sometimes be poor since exact entries are small
  tic
  for i = 1:m
    ei(FA.q(r(i,1))) = 1;
    ej(FA.p(r(i,2))) = 1;
    if dolu, S(i) = (FA.U'\ei)'*(FA.L\(FA.P*ej));
    else,    S(i) = (FA.L\(FA.P\ei))'*(FA.D\(FA.L\(FA.P\ej)));
    end
    ei(FA.q(r(i,1))) = 0;
    ej(FA.p(r(i,2))) = 0;
  end
  t = toc/m;  % average time
  for i = 1:m, T(i) = Y(r(i,1),i); end
  err = norm(S - T)/norm(T);
  fprintf('selinv err/avg time: %10.4e / %10.4e (s)\n',err,t)

  % diagonal inversion -- same as selinv but for most "important" entries
  tic
  for i = 1:m
    ei(FA.p(r(i,2))) = 1;
    if dolu, S(i) = (FA.U'\ei)'*(FA.L\(FA.P*ei));
    else,    S(i) = (FA.L\(FA.P\ei))'*(FA.D\(FA.L\(FA.P\ei)));
    end
    ei(FA.p(r(i,2))) = 0;
  end
  t = toc/m;  % average time
  for i = 1:m, T(i) = Y(r(i,2),i); end
  err = norm(S - T)/norm(T);
  fprintf('diaginv err/avg time: %10.4e / %10.4e (s)\n',err,t)
end

% kernel function
function K = Kfun(x,y,scale)
  dr = scale*abs(x' - y);  % scaled distance
  K = (1 + sqrt(3)*dr).*exp(-sqrt(3)*dr);
end

% matrix entries
function A = Afun_(i,j,x,noise,scale)
  A = Kfun(x(:,i),x(:,j),scale);
  [I,J] = ndgrid(i,j);
  idx = I == J;
  A(idx) = A(idx) + noise^2;  % modify diagonal with "nugget"
end

% proxy function
function [Kpxy,nbr] = pxyfun_(rc,rx,cx,slf,nbr,l,ctr,proxy,shift,scale)
  pxy = proxy + shift.*l + ctr;  % scale and translate reference points
  N = size(rx,2);
  if rc == 'r'
    Kpxy = Kfun(rx(:,slf),pxy,scale);
    dr = cx(:,nbr) - ctr;
  else
    Kpxy = Kfun(pxy,cx(:,slf),scale);
    dr = rx(:,nbr) - ctr;
  end
  % proxy points form "annulus" of scaled inner "radius" 1.5 around current box
  % keep among neighbors only those within annulus
  nbr = nbr(abs(dr)./l < 1.5);
end

% FFT multiplication
function y = mv_(F,x)
  n = length(x);
  y = ifft(F.*fft(x,2*n-1));
  y = y(1:n);
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