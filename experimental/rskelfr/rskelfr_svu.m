% RSKELFR_SVU  Solve by U factor in rectangular recursive skeletonization
%              factorization F = L*D*U.
%
%    See also RSKELFR, RSKELFR_SV.

function Y = rskelfr_svu(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'c'), ...
         'FLAM:rskelfr_svu:invalidTrans', ...
         'Transpose parameter must be either ''N'' or ''C''.')

  % initialize
  n = F.lvp(end);
  Y = X;

  % no transpose
  if strcmpi(trans,'n')
    for i = n:-1:1
      sk = F.factors(i).csk;
      rd = F.factors(i).crd;
      Y(rd,:) = Y(rd,:) - F.factors(i).F*Y(sk,:);
      Y(sk,:) = Y(sk,:) - F.factors(i).cT*Y(rd,:);
    end

  % conjugate transpose
  else
    for i = 1:n
      sk = F.factors(i).csk;
      rd = F.factors(i).crd;
      Y(rd,:) = Y(rd,:) - F.factors(i).cT'*Y(sk,:);
      Y(sk,:) = Y(sk,:) - F.factors(i).F'*Y(rd,:);
    end
  end
end