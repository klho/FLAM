% HIFIE_SPDIAG  Extract diagonal using hierarchical interpolative factorization
%               for integral equations via sparse apply/solves.
%
%    This algorithm computes each diagonal entry by multiplying from the left
%    and right by unit coordinate vectors, taking advantage of the sparsity of
%    each such operation.
%
%    Typical complexity: quasilinear in all dimensions, but generally worse than
%    HIFIE_DIAG. However, the constant can be significantly smaller so that this
%    should generally be preferred.
%
%    D = HIFIE_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = HIFIE_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_DIAG.

function D = hifie_spdiag(F,dinv)

  % set default parameters
  if nargin < 2 || isempty(dinv), dinv = 0; end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  n = F.lvp(end);
  spinfo.t = cell(n,1);  % block dependency tree for nonzero propagation
  x = zeros(1,N);        % current block for each index

  % bottom-up loop: set up immediate parent-child dependencies
  for lvl = 1:nlvl
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      f = F.factors(i);
      slf = [f.sk f.rd];

      % set ancestor dependencies of children blocks
      if lvl > 1
        chld = unique(x(slf));
        chld = chld(chld > 0);
        for j = chld, spinfo.t{j} = [spinfo.t{j} i]; end
      end

      % update block for each index
      x(slf) = i;
    end
  end

  % top-down loop: fill out full dependency tree
  for i = n:-1:1

    % fill out ancestry by copying from parent
    spinfo.t{i} = unique([i spinfo.t{spinfo.t{i}}]);

    % find leaf block for each index
    f = F.factors(i);
    x([f.sk f.rd]) = i;
  end

  % store leaf blocks and prune tree
  spinfo.i = unique(x);
  spinfo.t = spinfo.t(spinfo.i);

  % dispatch to eliminate overhead
  if F.symm == 'n'
    if dinv, D = hifie_spdiag_sv_n(F,spinfo);
    else,    D = hifie_spdiag_mv_n(F,spinfo);
    end
  elseif F.symm == 's'
    if dinv, D = hifie_spdiag_sv_s(F,spinfo);
    else,    D = hifie_spdiag_mv_s(F,spinfo);
    end
  elseif F.symm == 'h'
    if dinv, D = hifie_spdiag_sv_h(F,spinfo);
    else,    D = hifie_spdiag_mv_h(F,spinfo);
    end
  elseif F.symm == 'p'
    if dinv, D = hifie_spdiag_sv_p(F,spinfo);
    else,    D = hifie_spdiag_mv_p(F,spinfo);
    end
  end
end