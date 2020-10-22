% GAUSSPDF  Normal probability density function.
%
%    This is a compatibility wrapper for NORMPDF in case MATLAB's Statistics
%    Toolbox is not available.
%
%    Y = GAUSSPDF(X) returns the probability density function (PDF) at X of the
%    standard normal distribution.
%
%    Y = GAUSSPDF(X,MU) returns the PDF at X of the normal distribution with
%    mean MU and unit standard deviation.
%
%    Y = GAUSSPDF(X,MU,SIGMA) returns the PDF at X of the normal distribution
%    with mean MU and standard deviation SIGMA.

function y = gausspdf(x,mu,sigma)

  % set default parameters
  if nargin < 2, mu = 0; end
  if nargin < 3, sigma = 1; end

  % dispatch to native function or evaluate directly
  persistent e
  if isempty(e), e = exist('normpdf'); end
  if e, y = normpdf(x,mu,sigma);
  else, y = exp(-0.5*((x - mu)./sigma).^2)./(sqrt(2*pi)*sigma);
  end
end