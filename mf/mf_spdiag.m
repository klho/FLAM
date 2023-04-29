% MF_SPDIAG  Extract diagonal using multifrontal factorization via sparse
%            apply/solves.
%
%    This algorithm computes each diagonal entry by multiplying from the left
%    and right by unit coordinate vectors, taking advantage of the sparsity of
%    each such operation.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N*LOG^3(N)) in 1D and
%    O(N^(1 + 2*(1 - 1/D))) in D dimensions. This is generally worse than
%    MF_DIAG, but the constant can be substantially smaller.
%
%    D = MF_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = MF_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    See also MF2, MF3, MF_DIAG, MFX.

function D = mf_spdiag(F,dinv)

  % set default parameters
  if nargin < 2 || isempty(dinv), dinv = 0; end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  n = F.lvp(end);
  spinfo.t = cell(n,1);  % block dependency tree for nonzero propagation
  nbor = cell(n,1);      % neighbors for each block
  prnt = cell(n,1);      % "parents" for each block -- can be multiple
  x = cell(N,1);         % current blocks for each index -- can be multiple
  rem = true(1,N);       % which indices remain?

  % bottom-up loop: set up immediate parent-child and neighbor dependencies
  for lvl = 1:nlvl

    % initialize block indices
    x_ = x;         % store previous block data
    x = cell(N,1);  % reset current block data

    % loop through blocks on current level
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      f = F.factors(i);
      slf = [f.sk f.rd];

      if lvl == 1  % find neighbors if first elimination
        nbr = unique([x{slf}]);
        if ~isempty(nbr)
          for j = nbr
            nbor{i} = [nbor{i} j];
            nbor{j} = [nbor{j} i];
          end
        end
      else         % update parents of children blocks
        chld = unique([x_{slf}]);  % associated block indices at previous level
        for j = chld, prnt{j} = [prnt{j} i]; end
      end

      % update blocks for each index
      for j = slf, x{j} = [x{j} i]; end

      % remove redundant indices
      rem(f.rd) = false;
    end

    % pull block data from previous level if not touched at this level
    if lvl > 1
      for i = find(rem)
        if isempty(x{i}), x{i} = x_{i}; end
      end
    end
  end

  % top-down loop: fill out full dependency tree
  x = zeros(N,1);
  for i = n:-1:1

    % fill out ancestry by augmenting from parents and neighbors
    spinfo.t{i} = unique([i nbor{i} spinfo.t{prnt{i}}]);

    % find leaf block for each index
    f = F.factors(i);
    x([f.sk f.rd]) = i;
  end

  % store leaf blocks and prune tree
  spinfo.i = unique(x);
  spinfo.t = spinfo.t(spinfo.i);

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if dinv, D = mf_spdiag_sv_n(F,spinfo);
    else,    D = mf_spdiag_mv_n(F,spinfo);
    end
  elseif F.symm == 'h'
    if dinv, D = mf_spdiag_sv_h(F,spinfo);
    else,    D = mf_spdiag_mv_h(F,spinfo);
    end
  elseif F.symm == 'p'
    if dinv, D = mf_spdiag_sv_p(F,spinfo);
    else,    D = mf_spdiag_mv_p(F,spinfo);
    end
  end
end