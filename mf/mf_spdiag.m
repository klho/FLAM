% MF_SPDIAG  Extract diagonal using multifrontal factorization via sparse
%            apply/solves.
%
%    D = MF_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = MF_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    See also MF2, MF3, MF_DIAG, MFX.

function D = mf_spdiag(F,dinv)

  % set default parameters
  if nargin < 2 || isempty(dinv)
    dinv = 0;
  end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  n = F.lvp(end);
  spinfo.t = cell(n,1);
  nbor = cell(n,1);
  prnt = cell(n,1);
  rem = true(1,N);

  % build block dependency tree
  for lvl = 1:nlvl

    % initialize block indices
    if lvl > 1
      y = x;
    end
    x = cell(N,1);

    % loop through blocks on current level
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      slf = [sk rd];

      % find neighbors
      nbr = unique([x{slf}]);
      for j = nbr
        nbor{i} = [nbor{i} j];
        nbor{j} = [nbor{j} i];
      end

      % find parents
      if lvl > 1
        chld = unique([y{slf}]);
        for j = chld
          prnt{j} = [prnt{j} i];
        end
      end

      % update block for each index
      for j = slf
        x{j} = [x{j} i];
      end

      % remove redundant indices
      rem(rd) = false;
    end

    % pull block data from previous level if not touched at this level
    if lvl > 1
      for i = find(rem)
        if isempty(x{i})
          x{i} = y{i};
        end
      end
    end
  end

  % postprocess
  x = zeros(N,1);
  for i = n:-1:1

    % fill out ancestors for all blocks
    spinfo.t{i} = unique([i nbor{i} spinfo.t{prnt{i}}]);

    % find leaf block for each index
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    x([sk rd]) = i;
  end

  % store leaf blocks and subselect tree
  spinfo.i = unique(x);
  spinfo.t = spinfo.t(spinfo.i);

  % dispatch
  if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
    if dinv
      D = mf_spdiag_sv_n(F,spinfo);
    else
      D = mf_spdiag_mv_n(F,spinfo);
    end
  elseif strcmpi(F.symm,'h')
    if dinv
      D = mf_spdiag_sv_h(F,spinfo);
    else
      D = mf_spdiag_mv_h(F,spinfo);
    end
  elseif strcmpi(F.symm,'p')
    if dinv
      D = mf_spdiag_sv_p(F,spinfo);
    else
      D = mf_spdiag_mv_p(F,spinfo);
    end
  end
end