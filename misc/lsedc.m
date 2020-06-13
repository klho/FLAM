% LSEDC  Equality-constrained linear least squares using deferred correction.
%
%    X = LSEDC(LSFUN,A,B,C,D,TAU) produces the solution X to the equality-
%    constrained least squares problem
%
%      MIN. NORM(A*X - B)  S.T.  C*X = D
%
%    by solving the unconstrained weighted least squares problem
%
%      MIN. NORM(ATAU*X - BTAU),  ATAU = [TAU*C; A],  BTAU = [TAU*D; B],
%
%    where TAU is a fixed weighting factor, using deferred correction. The
%    argument LSFUN is a function to apply the pseudoinverse of ATAU.
%
%    [X,CRES] = LSEDC(LSFUN,A,B,C,D,TAU) also returns the constraint residual
%    matrix CRES upon termination.
%
%    [X,CRES] = LSEDC(LSFUN,A,B,C,D,TAU,TOL) iterates until the constraint
%    residual matrix has norm less than or equal to TOL (default: TOL = 1E-12).
%
%    [X,CRES] = LSEDC(LSFUN,A,B,C,D,TAU,TOL,NITER_MAX) uses at most NITER_MAX
%    iterations (default: NITER_MAX = 8).
%
%    [X,CRES,NITER] = LSEDC(LSFUN,A,B,C,D,TAU,...) further returns the number
%    NITER of iterations required for convergence. If NITER = -1, then
%    convergence was not detected.
%
%    References:
%
%      J.L. Barlow, U.B. Vemulapati. A note on deferred correction for equality
%        constrained least squares problems. SIAM J. Numer. Anal. 29 (1):
%        249-256, 1992.
%
%      C. Van Loan. On the method of weighting for equality-constrained least-
%        squares problems. SIAM J. Numer. Anal. 22 (5): 851-864, 1985.

function [x,w,niter] = lsedc(lsfun,A,B,C,D,tau,tol,niter_max)

  % set default parameters
  if nargin < 7 || isempty(tol), tol = 1e-12; end
  if nargin < 8 || isempty(niter_max), niter_max = 8; end

  % check inputs
  assert(tol >= 0,'FLAM:lsedc:invalidTol','Tolerance must be nonnegative.')
  assert(niter_max >= 0,'FLAM:lsedc:invalidMaxIter', ...
         'Maximum number of iterations must be nonnegative.')

  % initial solve
  x = lsfun([tau*D; B]);
  w = D - C*x;

  % return if all converged
  if norm(w) <= tol, niter = 0; return; end

  % iteratively correct constraints
  r = B - A*x;
  lambda = tau*w;
  for niter = 1:niter_max
    dx = lsfun([tau*w + lambda; r]);
    x = x + dx;
    w = w - C*dx;
    if norm(w) <= tol, return; end
    r = r - A*dx;
    lambda = lambda + tau*w;
  end

  % loop didn't return; no convergence
  niter = -1;
  warning('FLAM:lsedc:maxIterCount','Maximum number of iterations reached.')
end