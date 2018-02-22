% RSKELFR_MVD  Multiply by D factor in rectangular recursive skeletonization
%              factorization F = L*D*U.
%
%    See also RSKELFR, RSKELFR_MV.

function Y = rskelfr_mvd(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_mvd:invalidTrans', ...
         'Transpose parameter must be either ''N'' or ''C''.')

  % initialize
  n = F.lvp(end);

  % no transpose
  if strcmpi(trans,'n')
    Y = zeros(F.M,size(X,2));
    for i = 1:n
      rrd = F.factors(i).rrd;
      crd = F.factors(i).crd;
      Y(rrd,:) = F.factors(i).L*(F.factors(i).U*X(crd,:));
    end

  % conjugate transpose
  else
    Y = zeros(F.N,size(X,2));
    for i = 1:n
      rrd = F.factors(i).rrd;
      crd = F.factors(i).crd;
      Y(crd,:) = F.factors(i).U'*(F.factors(i).L'*X(rrd,:));
    end
  end
end