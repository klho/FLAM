% LSEDC  Equality-constrained linear least squares using deferred correction.
%
%    X = LSEDC(LSFUN,A,B,C,D,TAU) produces the solution X to the equality-
%    constrained least squares problem
%
%      MIN NORM(A*X - B)  S.T.  C*X = D
%
%    by solving the unconstrained weighted least squares problem
%
%      MIN NORM(ATAU*X - BTAU),  ATAU = [TAU*C; A],  BTAU = [TAU*D; B],
%
%    where TAU is a fixed weighting factor, using deferred correction. The
%    argument LSFUN is a function to apply the pseudoinverse of ATAU.
%
%    [X,CRES] = LSEDC(LSFUN,A,B,C,D,TAU) also returns the constraint residual
%    norm CRES upon termination.
%
%    [X,CRES,NITER] = LSEDC(LSFUN,A,B,C,D,TAU) further returns the number NITER
%    of deferred correction iterations required for convergence. If NITER = -1,
%    then convergence was not detected.
%
%    [X,CRES,NITER] = LSEDC(LSFUN,A,B,C,D,TAU,TOL) iterates until the constraint
%    residual has norm less than TOL (default: TOL = 1E-12).
%
%    [X,CRES,NITER] = LSEDC(LSFUN,A,B,C,D,TAU,TOL,NITER_MAX) uses at most
%    NITER_MAX iterations (default: NITER_MAX = 8).
%
%    References:
%
%      J.L. Barlow, U.B. Vemulapati. A note on deferred correction for equality
%        constrained least squares problems. SIAM J. Numer. Anal. 29 (1):
%        249-256, 1992.
%
%      C. Van Loan. On the method of weighting for equality-constrained least-
%        squares problems. SIAM J. Numer. Anal. 22 (5): 851-864, 1985.

function [x,cres,niter] = lsedc(lsfun,A,B,C,D,tau,tol,niter_max)

  % set default parameters
  if nargin < 7 || isempty(tol)
    tol = 1e-12;
  end
  if nargin < 8 || isempty(niter_max)
    niter_max = 8;
  end

  % check inputs
  assert(tol >= 0,'FLAM:lsedc:negativeTol','Tolerance must be nonnegative.')
  assert(niter_max > 0,'FLAM:lsedc:nonpositiveMaxIter', ...
         'Maximum number of iterations must be positive.')

  % initial solve
  x = lsfun([D; B]);
  r = B - A*x;
  w = D - C/tau*x;
  lambda = tau^2*w;

  % iteratively correct constraints
  for niter = 1:niter_max
    y = [tau*w + lambda/tau; r];
    dx = lsfun(y);
    x = x + dx;
    r = r - A*dx;
    w = w - C/tau*dx;
    lambda = lambda + tau^2*w;

    % return if all converged
    if all(sqrt(dot(w,w)) < tol)
      cres = norm(w);
      return
    end
  end

  % no convergence
  niter = -1;
  warning('FLAM:lsedc:maxIterCount','Maximum number of iterations reached.')
  cres = norm(w);
end