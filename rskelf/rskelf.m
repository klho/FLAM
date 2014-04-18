% RSKELF        Recursive skeletonization factorization.
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
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', and Hermitian if SYMM = 'H' (default:
%              SYMM = 'N'). If SYMM = 'N' or 'S', then local factors are
%              computed using the LU decomposition; if SYMM = 'H', then these
%              are computed using the LDL decomposition.
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      A. Gillman, P.M. Young, P.-G. Martinsson. A direct solver with O(N)
%        complexity for integral equations on one-dimensional domains. Front.
%        Math. China 7 (2): 217-247, 2012.
%
%      K.L. Ho, L. Greengard. A fast direct solver for structured linear systems
%        by recursive skeletonization. SIAM J. Sci. Comput. 34 (5): A2507-A2532,
%        2012.
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. arXiv:1307.2666, 2013.
%
%      P.G. Martinsson, V. Rokhlin. A fast direct solver for boundary integral
%        equations in two dimensions. J. Comput. Phys. 205: 1-23, 2005.
%
%    See also HYPOCT, ID, RSKELF_MV, RSKELF_SV.

function F = rskelf(A,x,occ,rank_or_tol,pxyfun,opts)
  start = tic;

  % set default parameters
  if nargin < 5
    pxyfun = [];
  end
  if nargin < 6
    opts = [];
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
  opts.symm = lower(opts.symm);
  if ~(strcmp(opts.symm,'n') || strcmp(opts.symm,'s') || strcmp(opts.symm,'h'))
    error('FLAM:rskelf:invalidSymm', ...
          'Symmetry parameter must be one of ''N'', ''S'', or ''H''.')
  end

  % build tree
  N = size(x,2);
  tic
  t = hypoct(x,occ,opts.lvlmax);

  % print summary
  if opts.verb
    fprintf(['-'*ones(1,80) '\n'])
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
  F = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'P',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',t.nlvl,'lvp',zeros(1,t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = ones(N,1);
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

      % compute interaction matrices
      K = full(A(nbr,slf));
      if strcmp(opts.symm,'n')
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
      if strcmp(opts.symm,'n') || strcmp(opts.symm,'h')
        K(rd,:) = K(rd,:) - T'*K(sk,:);
      elseif strcmp(opts.symm,'s')
        K(rd,:) = K(rd,:) - T.'*K(sk,:);
      end
      K(:,rd) = K(:,rd) - K(:,sk)*T;
      if strcmp(opts.symm,'n') || strcmp(opts.symm,'s')
        [L,U,P] = lu(K(rd,rd));
        X = U\(L\P);
      elseif strcmp(opts.symm,'h')
        [L,U,P] = ldl(K(rd,rd));
        X = P*(L'\(U\(L\P')));
      end
      E = K(sk,rd)*X;
      if strcmp(opts.symm,'n')
        G = X*K(rd,sk);
      elseif strcmp(opts.symm,'s') || strcmp(opts.symm,'h')
        G = [];
      end

      % update self-interaction
      M{i} = M{i}(sk,sk) - K(sk,rd)*X*K(rd,sk);

      % store matrix factors
      n = n + 1;
      F.factors(n).sk = slf(sk);
      F.factors(n).rd = slf(rd);
      F.factors(n).T = T;
      F.factors(n).E = E;
      F.factors(n).F = G;
      F.factors(n).P = P;
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
    fprintf(['-'*ones(1,80) '\n'])
    toc(start)
  end
end