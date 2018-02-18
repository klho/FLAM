% RSKELF   Recursive skeletonization factorization.
%
%    F = RSKELF(A,X,OCC,RANK_OR_TOL,PXYFUN) produces a factorization F of the
%    interaction matrix A on the points X using tree occupancy parameter OCC,
%    local precision parameter RANK_OR_TOL, and proxy function PXYFUN to capture
%    the far field. This is a function of the form
%
%      [KPXY,NBR] = PXYFUN(X,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor indices (can be modified)
%      - X:    input points
%      - SLF:  block indices
%      - L:    block size
%      - CTR:  block center
%
%    See the examples for further details. If PXYFUN is not provided or empty
%    (default), then the code uses the naive global compression scheme.
%
%    F = RSKELF(A,X,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options to
%    the algorithm. Valid options include:
%
%      - EXT: set the root node extent to [EXT(I,1) EXT(I,2)] along dimension I.
%             If EXT is empty (default), then the root extent is calculated from
%             the data.
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N'). If
%              SYMM = 'N' or 'S', then local factors are computed using the LU
%              decomposition; if SYMM = 'H', the LDL decomposition; and if
%              SYMM = 'P', the Cholesky decomposition.
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      S. Chandrasekaran, M. Gu, T. Pals. A fast ULV decomposition solver for
%        hierarchically semiseparable representations. SIAM J. Matrix Anal.
%        Appl. 28 (3): 603-622, 2006.
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. Comm. Pure Appl. Math. 69 (7):
%        1314-1353, 2016.
%
%      J. Xia, S. Chandrasekaran, M. Gu, X.S. Li. Fast algorithms for
%        hierarchically semiseparable matrices. Numer. Linear Algebra Appl. 17
%        (6): 953-976, 2010.
%
%    See also HYPOCT, ID, RSKELF_CHOLMV, RSKELF_CHOLSV, RSKELF_DIAG,
%    RSKELF_LOGDET, RSKELF_MV, RSKELF_SPDIAG, RSKELF_SV.

function F = rskelf(A,x,occ,rank_or_tol,pxyfun,opts)
  start = tic;

  % set default parameters
  if nargin < 5
    pxyfun = [];
  end
  if nargin < 6
    opts = [];
  end
  if ~isfield(opts,'ext')
    opts.ext = [];
  end
  if ~isfield(opts,'lvlmax')
    opts.lvlmax = Inf;
  end
  if ~isfield(opts,'symm')
    opts.symm = 'n';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:rskelf:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')
  if strcmpi(opts.symm,'h') && isoctave()
    warning('FLAM:rskelf:octaveLDL','No LDL decomposition in Octave; using LU.')
    opts.symm = 's';
  end

  % build tree
  N = size(x,2);
  tic
  t = hypoct(x,occ,opts.lvlmax,opts.ext);

  % print summary
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    fprintf('%3s | %63.2e (s)\n','-',toc)

    % count nonempty boxes at each level
    pblk = zeros(t.nlvl+1,1);
    for lvl = 1:t.nlvl
      pblk(lvl+1) = pblk(lvl);
      for i = t.lvp(lvl)+1:t.lvp(lvl+1)
        if ~isempty(t.nodes(i).xi)
          pblk(lvl+1) = pblk(lvl+1) + 1;
        end
      end
    end
  end

  % initialize
  nbox = t.lvp(end);
  e = cell(nbox,1);
  F = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',t.nlvl,'lvp',zeros(1,t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = true(N,1);
  M = cell(nbox,1);
  I = zeros(N,1);

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    tic
    nlvl = nlvl + 1;
    nrem1 = sum(rem);
    l = t.lrt/2^(lvl - 1);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).xi = [t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]];
    end

    % loop over nodes
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      slf = t.nodes(i).xi;
      nbr = [t.nodes(t.nodes(i).nbor).xi];
      nslf = length(slf);

      % generate modified diagonal block
      M{i} = full(A(slf,slf));
      if lvl < t.nlvl
        I(slf) = 1:nslf;
        for j = t.nodes(i).chld
          xi = t.nodes(j).xi;
          M{i}(I(xi),I(xi)) = M{j};
          M{j} = [];
        end
      end

      % compute proxy interactions and subselect neighbors
      Kpxy = zeros(0,nslf);
      if lvl > 2
        if isempty(pxyfun)
          nbr = setdiff(find(rem),slf);
        else
          [Kpxy,nbr] = pxyfun(x,slf,nbr,l,t.nodes(i).ctr);
        end
      end

      % compute interaction matrix
      K = full(A(nbr,slf));
      if strcmpi(opts.symm,'n')
        K = [K; full(A(slf,nbr))'];
      end
      K = [K; Kpxy];

      % skeletonize
      [sk,rd,T] = id(K,rank_or_tol);

      % move on if no compression
      if isempty(rd)
        continue
      end

      % compute factors
      K = M{i};
      if strcmpi(opts.symm,'s')
        K(rd,:) = K(rd,:) - T.'*K(sk,:);
      else
        K(rd,:) = K(rd,:) - T'*K(sk,:);
      end
      K(:,rd) = K(:,rd) - K(:,sk)*T;
      if strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s')
        [L,U] = lu(K(rd,rd));
        E = K(sk,rd)/U;
        G = L\K(rd,sk);
      elseif strcmpi(opts.symm,'h')
        [L,U] = ldl(K(rd,rd));
        E = (K(sk,rd)/L')/U;
        G = [];
      elseif strcmpi(opts.symm,'p')
        L = chol(K(rd,rd),'lower');
        U = [];
        E = K(sk,rd)/L';
        G = [];
      end

      % update self-interaction
      if strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s')
        M{i} = M{i}(sk,sk) - E*G;
      elseif strcmpi(opts.symm,'h')
        M{i} = M{i}(sk,sk) - E*U*E';
      elseif strcmpi(opts.symm,'p')
        M{i} = M{i}(sk,sk) - E*E';
      end

      % store matrix factors
      n = n + 1;
      F.factors(n).sk = slf(sk);
      F.factors(n).rd = slf(rd);
      F.factors(n).T = T;
      F.factors(n).E = E;
      F.factors(n).F = G;
      F.factors(n).L = L;
      F.factors(n).U = U;

      % restrict to skeletons
      t.nodes(i).xi = slf(sk);
      rem(slf(rd)) = 0;
    end
    F.lvp(nlvl+1) = n;

    % print summary
    if opts.verb
      nrem2 = sum(rem);
      nblk = pblk(lvl) + t.lvp(lvl+1) - t.lvp(lvl);
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
              lvl,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,toc)
    end
  end

  % finish
  F.factors = F.factors(1:n);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end