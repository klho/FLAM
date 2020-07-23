% HIFIE_SPDIAG_MV_H  Dispatch for HIFIE_SPDIAG with DINV = 0 and F.SYMM = 'H'.

function D = hifie_spdiag_mv_h(F,spinfo)

  % initialize
  N = F.N;
  n = length(spinfo.i);
  P = zeros(N,1);  % for indexing
  D = zeros(N,1);  % for output

  % loop over all leaf blocks from top to bottom
  for i = n:-1:1

    % find active indices for current block
    rem = spinfo.t{i};
    rem = unique([[F.factors(rem).sk] [F.factors(rem).rd]]);
    nrem = length(rem);
    P(rem) = 1:nrem;

    % allocate active submatrix for current block
    j = spinfo.i(i);
    sk = F.factors(j).sk;
    rd = F.factors(j).rd;
    slf = [sk rd];
    nslf = length(slf);
    Y = zeros(nrem,nslf);
    Y(P(slf),:) = eye(nslf);

    % upward sweep
    for j = spinfo.t{i}
      sk = P(F.factors(j).sk);
      rd = P(F.factors(j).rd);
      Y(sk,:) = Y(sk,:) + F.factors(j).T*Y(rd,:);
      Y(rd,:) = F.factors(j).L'*Y(rd,:);
      Y(rd,:) = Y(rd,:) + F.factors(j).E'*Y(sk,:);
    end

    % store matrix at top level
    Z = Y;

    % apply diagonal factors
    for j = spinfo.t{i}
      rd = P(F.factors(j).rd);
      Y(rd,:) = F.factors(j).U*Y(rd,:);
    end

    % extract diagonal
    D(slf) = diag(Z'*Y);
  end
end