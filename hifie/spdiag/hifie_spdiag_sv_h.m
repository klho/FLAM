% HIFIE_SPDIAG_SV_H  Dispatch for HIFIE_SPDIAG with DINV = 1 and F.SYMM = 'H'.

function D = hifie_spdiag_sv_h(F,spinfo)

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
    f = F.factors(spinfo.i(i));
    slf = [f.sk f.rd];
    nslf = length(slf);
    Y = zeros(nrem,nslf);
    Y(P(slf),:) = eye(nslf);

    % upward sweep
    for j = spinfo.t{i}
      f = F.factors(j);
      sk = P(f.sk); rd = P(f.rd);
      Y(rd,:) = Y(rd,:) - f.T'*Y(sk,:);
      Y(rd,:) = f.L\Y(rd,:);
      Y(sk,:) = Y(sk,:) - f.E*Y(rd,:);
    end

    % store matrix at top level
    Z = Y;

    % apply diagonal factors
    for j = spinfo.t{i}
      f = F.factors(j);
      rd = P(f.rd);
      Y(rd,:) = f.U\Y(rd,:);
    end

    % extract diagonal
    D(slf) = diag(Z'*Y);
  end
end