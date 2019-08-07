% MF_SPDIAG_SV_N  Dispatch for MF_SPDIAG with DINV = 1 and F.SYMM = 'N'.

function D = mf_spdiag_sv_n(F,spinfo)

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
    nrem = length(rem);  % total storage needed
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
      Y(rd,:) = F.factors(j).L\Y(rd(F.factors(j).p),:);
      Y(sk,:) = Y(sk,:) - F.factors(j).E*Y(rd,:);
    end

    % downward sweep
    for j = spinfo.t{i}(end:-1:1)
      sk = P(F.factors(j).sk);
      rd = P(F.factors(j).rd);
      Y(rd,:) = Y(rd,:) - F.factors(j).F*Y(sk,:);
      Y(rd,:) = F.factors(j).U\Y(rd,:);
    end

    % extract diagonal
    D(slf) = diag(Y(P(slf),:));
  end
end