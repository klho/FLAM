% SNORM  Spectral norm estimation using randomized power method.
%
%    S = SNORM(N,MV,MVA) produces a spectral norm estimate S of a matrix with
%    column dimension N and functions MV and MVA to apply the matrix and its
%    adjoint, respectively, to a vector using the randomized power method. If A
%    is the matrix of interest, then the multiplication functions must satisfy
%    A*X = MV(X) and A'*X = MVA(X).
%
%    S = SNORM(N,MV,MVA,TOL) estimates the spectral norm to relative precision
%    TOL (default: TOL = 1E-2).
%
%    S = SNORM(N,MV,MVA,TOL,HERM) optimizes for Hermiticity if HERM = 1
%    (default: HERM = 0), in which case MVA is not used and the number of
%    multiplications is halved.
%
%    S = SNORM(N,MV,MVA,TOL,HERM,NITER_MAX) uses at most NITER_MAX iterations
%    (default: NITER_MAX = 32).
%
%    [S,NITER] = SNORM(N,MV,MVA,...) also returns the number NITER of iterations
%    required for convergence. If NITER = -1, then convergence was not detected.
%
%    References:
%
%      J.D. Dixon. Estimating extremal eigenvalues and condition numbers of
%        matrices. SIAM J. Numer. Anal. 20(4): 812-814, 1983.
%
%      J. Kuczynski, H. Wozniakowski. Estimating the largest eigenvalue by the
%        power and Lanczos algorithms with a random start. SIAM J. Matrix Anal.
%        Appl. 13(4): 1094-1122, 1992.

function [s,niter] = snorm(n,mv,mva,tol,herm,niter_max)

  % set default parameters
  if nargin < 4 || isempty(tol), tol = 1e-2; end
  if nargin < 5 || isempty(herm), herm = 0; end
  if nargin < 6 || isempty(niter_max), niter_max = 32; end

  % check inputs
  assert(tol >= 0,'FLAM:snorm:invalidTol','Tolerance must be nonnegative.')
  assert(niter_max > 0,'FLAM:snorm:invalidMaxIter', ...
         'Maximum number of iterations must be positive.')

  % initialize
  x = rand(n,1);
  xnorm = norm(x);
  s = xnorm;

  % main loop
  for niter = 1:niter_max
    x = x/xnorm;
    s_ = s;

    % apply matrix
    if herm, x = mv(x);
    else,    x = mva(mv(x));
    end
    xnorm = norm(x);

    % estimate spectral norm
    if herm, s = xnorm;
    else,    s = sqrt(xnorm);
    end

    % check for termination
    if xnorm <= eps, return; end  % avoid divide by zero
    if abs(s - s_) <= s_*tol, return; end  % within tolerance
  end

  % loop didn't return; no convergence
  niter = -1;
  warning('FLAM:snorm:maxIterCount','Maximum number of iterations reached.')
end