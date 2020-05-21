% Five-point stencil on the unit square, Poisson equation.
%
% This example solves the Poisson equation on the unit square with Dirichlet
% boundary conditions. The system is discretized using the standard five-point
% stencil; the resulting matrix is square, real, and positive definite.
%
% This demo does the following in order:
%
%   - factor the matrix
%   - check multiply/solve error/time
%   - compare CG with/without preconditioning by approximate solve
%
% Inputs (defaults are used if not provided or set empty):
%
%   - N: number of discretization points in each dimension (default: N = 128)
%   - OCC: tree occupancy parameter (default: OCC = 64)
%   - RANK_OR_TOL: local precision parameter (default: RANK_OR_TOL = 1e-9)
%   - TMAX: ID interpolation matrix entry bound (default: TMAX = 2)
%   - SYMM: symmetry parameter (default: SYMM = 'P')
%   - DOITER: whether to run unpreconditioned CG (default: DOITER = 1)

function fd_square(n,occ,rank_or_tol,Tmax,symm,doiter)

  % set default parameters
  if nargin < 1 || isempty(n), n = 128; end
  if nargin < 2 || isempty(occ), occ = 64; end
  if nargin < 3 || isempty(rank_or_tol), rank_or_tol = 1e-9; end
  if nargin < 4 || isempty(Tmax), Tmax = 2; end
  if nargin < 5 || isempty(symm), symm = 'p'; end
  if nargin < 6 || isempty(doiter), doiter = 1; end

  % initialize
  [x1,x2] = ndgrid((1:n)/n); x = [x1(:) x2(:)]'; clear x1 x2  % grid points
  N = size(x,2);

  % set up sparse matrix
  idx = reshape(1:N,n,n);  % index mapping to each point
  % interaction with middle (self)
  Im = idx(1:n,1:n); Jm = idx(1:n,1:n); Sm = 4*ones(size(Im));
  % interaction with left
  Il = idx(1:n-1,1:n); Jl = idx(2:n,1:n); Sl = -ones(size(Il));
  % interaction with right
  Ir = idx(2:n,1:n); Jr = idx(1:n-1,1:n); Sr = -ones(size(Ir));
  % interaction with up
  Iu = idx(1:n,1:n-1); Ju = idx(1:n,2:n); Su = -ones(size(Iu));
  % interaction with down
  Id = idx(1:n,2:n); Jd = idx(1:n,1:n-1); Sd = -ones(size(Id));
  % combine all interactions
  I = [Im(:); Il(:); Ir(:); Iu(:); Id(:)];
  J = [Jm(:); Jl(:); Jr(:); Ju(:); Jd(:)];
  S = [Sm(:); Sl(:); Sr(:); Su(:); Sd(:)];
  A = sparse(I,J,S,N,N);
  clear idx Im Jm Sm Il Jl Sl Ir Jr Sr Iu Ju Su Id Jd Sd I J S

  % factor matrix
  Afun = @(i,j)spget(A,i,j);
  pxyfun = @(x,slf,nbr,l,ctr)pxyfun_(x,slf,nbr,l,ctr,A);
  opts = struct('Tmax',Tmax,'symm',symm,'verb',1);
  tic; F = rskelf(Afun,x,occ,rank_or_tol,pxyfun,opts); t = toc;
  w = whos('F'); mem = w.bytes/1e6;
  fprintf('rskelf time/mem: %10.4e (s) / %6.2f (MB)\n',t,mem)

  % test accuracy using randomized power method
  X = rand(N,1);
  X = X/norm(X);

  % NORM(A - F)/NORM(A)
  tic; rskelf_mv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(A*x - rskelf_mv(F,x)),[],[],1);
  err = err/snorm(N,@(x)(A*x),[],[],1);
  fprintf('rskelf_mv err/time: %10.4e / %10.4e (s)\n',err,t)

  % NORM(INV(A) - INV(F))/NORM(INV(A)) <= NORM(I - A*INV(F))
  tic; rskelf_sv(F,X); t = toc;  % for timing
  err = snorm(N,@(x)(x - A*rskelf_sv(F,x)),@(x)(x - rskelf_sv(F,A*x,'c')));
  fprintf('rskelf_sv err/time: %10.4e / %10.4e (s)\n',err,t)

  % run unpreconditioned CG
  B = A*X;
  iter = nan;
  if doiter, [~,~,~,iter] = pcg(@(x)(A*x),B,1e-12,128); end

  % run preconditioned CG
  tic; [Y,~,~,piter] = pcg(@(x)(A*x),B,1e-12,32,@(x)rskelf_sv(F,x)); t = toc;
  err1 = norm(X - Y)/norm(X);
  err2 = norm(B - A*Y)/norm(B);
  fprintf('cg:\n')
  fprintf('  soln/resid err/time: %10.4e / %10.4e / %10.4e (s)\n', ...
          err1,err2,t)
  fprintf('  precon/unprecon iter: %d / %d\n',piter,iter)
end

% proxy function
function [Kpxy,nbr] = pxyfun_(x,slf,nbr,l,ctr,A)
  % only neighbor interactions -- no far field
  Kpxy = zeros(0,length(slf));
  % keep only neighbors with nonzero interaction
  nbr = sort(nbr);
  [I,~] = find(A(:,slf)); I = unique(I);
  nbr = nbr(ismemb(nbr,I));
end