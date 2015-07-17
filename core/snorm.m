% SNORM  Estimate spectral norm using the randomized power method.
%
%    S = SNORM(N,MV,MVA) produces a spectral norm estimate S of a matrix with
%    column dimension N and functions MV and MVA to apply the matrix and its
%    adjoint, respectively, to a vector using the randomized power method.
%
%    S = SNORM(N,MV,MVA,NITER) uses NITER steps of the randomized power method.
%
%    S = SNORM(N,MV,MVA,TOL) estimates the spectral norm to relative precision
%    TOL.
%
%    S = SNORM(N,MV,MVA,NITER_OR_TOL) sets NITER = NITER_OR_TOL if
%    NITER_OR_TOL >= 1 and TOL = NITER_OR_TOL if NITER_OR_TOL < 1 (default:
%    NITER_OR_TOL = 1E-2).
%
%    S = SNORM(N,MV,MVA,NITER_OR_TOL,HERM) uses symmetry optimizations if
%    HERM = 1 (the function MVA is not used and the number of multiplications is
%    halved).
%
%    S = SNORM(N,MV,MVA,TOL,HERM,NITER_MAX) estimates the spectral norm to
%    precision TOL using at most NITER_MAX iterations (default: NITER_MAX = 32).
%
%    References:
%
%      J.D. Dixon. Estimating extremal eigenvalues and condition numbers of
%        matrices. SIAM J. Numer. Anal. 20(4): 812-814, 1983.
%
%      J. Kuczynski, H. Wozniakowski. Estimating the largest eigenvalue by the
%        power and Lanczos algorithms with a random start. SIAM J. Matrix Anal.
%        Appl. 13(4): 1094-1122, 1992.

function [s,niter] = snorm(n,mv,mva,niter_or_tol,herm,niter_max)

  % set default parameters
  if nargin < 4 || isempty(niter_or_tol)
    niter_or_tol = 1e-2;
  end
  if nargin < 5 || isempty(herm)
    herm = 0;
  end
  if nargin < 6 || isempty(niter_max)
    niter_max = 32;
  end

  % check inputs
  assert(niter_or_tol >= 0,'FLAM:snorm:negativeIterOrTol', ...
         'Number of iterations or tolerance must be nonnegative.')
  assert(niter_max > 0,'FLAM:snorm:nonpositiveMaxIter', ...
         'Maximum number of iterations must be positive.')

  % initialize
  x = rand(n,1);
  nrm = sqrt(dot(x,x));
  s = nrm;
  niter = 0;

  % run power method
  while 1
    niter = niter + 1;
    x = bsxfun(@rdivide,x,nrm);
    s_ = s;

    % apply matrix
    if herm
      x = mv(x);
    else
      x = mva(mv(x));
    end
    nrm = sqrt(dot(x,x));

    % estimate spectral norm
    if herm
      s = nrm;
    else
      s = sqrt(nrm);
    end

    % check for termination
    if nrm > eps
      if niter_or_tol < 1
        if all(abs(s - s_) > abs(s_)*niter_or_tol) && niter < niter_max
          continue
        end
      else
        if niter < niter_or_tol
          continue
        end
      end
    end

    % check if maximum number of iterations reached
    if niter_or_tol < 1 && niter == niter_max
      warning('FLAM:snorm:maxIterCount','Maximum number of iterations reached.')
    end
    return
  end
end