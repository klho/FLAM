% RSKELFR_SVD  Solve by D factor in range-restricted recursive skeletonization
%              factorization F = L*D*U.
%
%    See also RSKELFR, RSKELFR_SV.

function Y = rskelfr_svd(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_svd:invalidTrans', ...
         'Transpose parameter must be either ''N'' or ''C''.')

  % initialize
  n = F.lvp(end);

  % no transpose
  if strcmpi(trans,'n')
    Y = zeros(F.N,size(X,2));
    for i = 1:n
      rrd = F.factors(i).rrd;
      crd = F.factors(i).crd;
      nrrd = length(rrd);
      ncrd = length(crd);
      if nrrd > ncrd
        Y(crd,:) = F.factors(i).U\(F.factors(i).L'*X(rrd,:));
      elseif nrrd < ncrd
        Y(crd,:) = F.factors(i).U'*(F.factors(i).L\X(rrd,:));
      else
        Y(crd,:) = F.factors(i).U\(F.factors(i).L\X(rrd,:));
      end
    end

  % conjugate transpose
  else
    Y = zeros(F.M,size(X,2));
    for i = 1:n
      rrd = F.factors(i).rrd;
      crd = F.factors(i).crd;
      nrrd = length(rrd);
      ncrd = length(crd);
      if nrrd > ncrd
        L = F.factors(i).U';
        Y(rrd,:) = F.factors(i).L*(L\X(crd,:));
      elseif nrrd < ncrd
        U = F.factors(i).L';
        Y(rrd,:) = U\(F.factors(i).U*X(crd,:));
      else
        L = F.factors(i).U';
        U = F.factors(i).L';
        Y(rrd,:) = U\(L\X(crd,:));
      end
    end
  end
end