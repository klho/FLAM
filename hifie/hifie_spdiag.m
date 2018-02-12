% HIFIE_SPDIAG   Extract diagonal using hierarchical interpolative factorization
%                for integral equations via sparse apply/solves.
%
%    D = HIFIE_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = HIFIE_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_DIAG.

function D = hifie_spdiag(F,dinv)

  % set default parameters
  if nargin < 2 || isempty(dinv)
    dinv = 0;
  end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  n = F.lvp(end);
  spinfo.t = cell(n,1);
  x = zeros(1,N);

  % build block dependency tree
  for lvl = 1:nlvl
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      slf = [sk rd];

      % set ancestor dependencies of children blocks
      if lvl > 1
        chld = unique(x(slf));
        chld = chld(chld > 0);
        for j = chld
          spinfo.t{j} = [spinfo.t{j} i];
        end
      end

      % update block for each index
      x(slf) = i;
    end
  end

  % postprocess
  for i = n:-1:1

    % fill out ancestors for all blocks
    spinfo.t{i} = unique([i spinfo.t{spinfo.t{i}}]);

    % find leaf block for each index
    sk = F.factors(i).sk;
    rd = F.factors(i).rd;
    x([sk rd]) = i;
  end

  % store leaf blocks and subselect tree
  spinfo.i = unique(x);
  spinfo.t = spinfo.t(spinfo.i);

  % dispatch
  if strcmpi(F.symm,'n')
    if dinv
      D = hifie_spdiag_sv_n(F,spinfo);
    else
      D = hifie_spdiag_mv_n(F,spinfo);
    end
  elseif strcmpi(F.symm,'s')
    if dinv
      D = hifie_spdiag_sv_s(F,spinfo);
    else
      D = hifie_spdiag_mv_s(F,spinfo);
    end
  elseif strcmpi(F.symm,'h')
    if dinv
      D = hifie_spdiag_sv_h(F,spinfo);
    else
      D = hifie_spdiag_mv_h(F,spinfo);
    end
  elseif strcmpi(F.symm,'p')
    if dinv
      D = hifie_spdiag_sv_p(F,spinfo);
    else
      D = hifie_spdiag_mv_p(F,spinfo);
    end
  end
end