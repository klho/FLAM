% HIFDE_SPDIAG_MV_S  Dispatch for HIFDE_SPDIAG with DINV = 0 and F.SYMM = 'S'.

function D = hifde_spdiag_mv_s(F,spinfo)

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
    f = F.factors(spinfo.i(i));
    slf = [f.sk f.rd];
    nslf = length(slf);
    Y = zeros(nrem,nslf);
    Y(P(slf),:) = eye(nslf);

    % upward sweep
    for j = spinfo.t{i}
      f = F.factors(j);
      sk = P(f.sk); rd = P(f.rd);
      if ~isempty(f.T), Y(sk,:) = Y(sk,:) + f.T*Y(rd,:); end
      Y(rd,:) = f.U*Y(rd,:);
      Y(rd,:) = Y(rd,:) + f.F*Y(sk,:);
    end

    % downward sweep
    for j = spinfo.t{i}(end:-1:1)
      f = F.factors(j);
      sk = P(f.sk); rd = P(f.rd);
      Y(sk,:) = Y(sk,:) + f.E*Y(rd,:);
      Y(rd(f.p),:) = f.L*Y(rd,:);
      if ~isempty(f.T), Y(rd,:) = Y(rd,:) + f.T.'*Y(sk,:); end
    end

    % extract diagonal
    D(slf) = diag(Y(P(slf),:));
  end
end