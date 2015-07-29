% HIFDE_SPDIAG_MV_N  Dispatch for HIFDE_SPDIAG with DINV = 0 and F.SYMM = 'N'.

function D = hifde_spdiag_mv_n(F,spinfo)

  % initialize
  N = F.N;
  n = length(spinfo.i);
  P = zeros(N,1);
  D = zeros(N,1);

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
      if j > 0
        sk = P(F.factors(j).sk);
        rd = P(F.factors(j).rd);
        T = F.factors(j).T;
        if ~isempty(T)
          Y(sk,:) = Y(sk,:) + T*Y(rd,:);
        end
        Y(rd,:) = F.factors(j).U*Y(rd,:);
        Y(rd,:) = Y(rd,:) + F.factors(j).F*Y(sk,:);
      end
    end

    % downward sweep
    for j = spinfo.t{i}(end:-1:1)
      if j > 0
        sk = P(F.factors(j).sk);
        rd = P(F.factors(j).rd);
        T = F.factors(j).T;
        Y(sk,:) = Y(sk,:) + F.factors(j).E*Y(rd,:);
        Y(rd,:) = F.factors(j).L*Y(rd,:);
        if ~isempty(T)
          Y(rd,:) = Y(rd,:) + T'*Y(sk,:);
        end
      end
    end

    % extract diagonal
    D(slf) = diag(Y(P(slf),:));
  end
end