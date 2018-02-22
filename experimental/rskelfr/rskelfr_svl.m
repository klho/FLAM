% RSKELFR_SVL  Solve by L factor in rectangular recursive skeletonization
%              factorization F = L*D*U.
%
%    See also RSKELFR, RSKELFR_SV.

function Y = rskelfr_svl(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_svl:invalidTrans', ...
         'Transpose parameter must be either ''N'' or ''C''.')

  % initialize
  n = F.lvp(end);
  Y = X;

  % no transpose
  if strcmpi(trans,'n')
    for i = 1:n
      sk = F.factors(i).rsk;
      rd = F.factors(i).rrd;
      Y(rd,:) = Y(rd,:) - F.factors(i).rT*Y(sk,:);
      Y(sk,:) = Y(sk,:) - F.factors(i).E*Y(rd,:);
    end

  % conjugate transpose
  else
    for i = n:-1:1
      sk = F.factors(i).rsk;
      rd = F.factors(i).rrd;
      Y(rd,:) = Y(rd,:) - F.factors(i).E'*Y(sk,:);
      Y(sk,:) = Y(sk,:) - F.factors(i).rT'*Y(rd,:);
    end
  end
end