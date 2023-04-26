% LOGDET_LDL  Compute log-determinant of D factor from block LDL factorization.
%
%    LD = LOGDET_LDL(D) produces the log-determinant LD = LOG(DET(D)).

function ld = logdet_ldl(D)
  idx = full(sum(D ~= 0)) == 1;  % find 1x1 blocks ...
  d = diag(D);                   % ... and compute determinant ...
  ld = sum(log(d(idx)));         % ... from corresponding diagonal
  idx = find(~idx);  % accumulate determinant for each 2x2 block
  for i = 1:2:length(idx), ld = ld + log(det(D(idx(i:i+1),idx(i:i+1)))); end
end