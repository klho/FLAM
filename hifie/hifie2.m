% HIFIE2  Hierarchical interpolative factorization for integral equations in 2D.
%
%    The hierarchical interpolative factorization is an extension of the
%    recursive skeletonization factorization with additional dimensional
%    reduction to achieve improved complexities in 2D and above. Specifically,
%    in 2D, it compresses cells to edges as in RSKELF, then further compresses
%    edges to points. The output is an approximation of a hierarchically rank-
%    structured matrix A as a generalized LDU (or LDL/Cholesky) decomposition.
%
%    This implementation is suited for first-kind integral equations or matrices
%    that otherwise do not have high contrast between the diagonal and off-
%    diagonal elements. For second-kind integral equations, see HIFIE2X.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N) in all dimensions.
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL) produces a factorization F of the matrix A
%    acting on the points X using tree occupancy parameter OCC and local
%    precision parameter RANK_OR_TOL. See HYPOCT and ID for details. Since no
%    proxy function is supplied, this simply performs a naive compression of all
%    off-diagonal blocks.
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL,PXYFUN) accelerates the compression using
%    the proxy function PXYFUN to capture the far field (both incoming and
%    outgoing). This is a function of the form
%
%      [KPXY,NBR] = PXYFUN(X,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor point indices (can be modified)
%      - X:    input points
%      - SLF:  block point indices
%      - L:    block node size
%      - CTR:  block node center
%
%    The relevant arguments will be passed in by the algorithm; the user is
%    responsible for handling them. See the examples for further details. If
%    PXYFUN is not provided or empty (default), then the code uses the naive
%    global compression scheme.
%
%    F = HIFIE2(A,X,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options to
%    the algorithm. Valid options include:
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf). See HYPOCT.
%
%      - EXT: set the root node extent to [EXT(D,1) EXT(D,2)] along dimension D.
%             If EXT is empty (default), then the root extent is calculated from
%             the data. See HYPOCT.
%
%      - SKIP: skip the dimension reductions on the first SKIP levels (default:
%              SKIP = 0).
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N'). If
%              SYMM = 'N' or 'S', then local factors are computed using the LU
%              decomposition; if SYMM = 'H', the LDL decomposition; and if
%              SYMM = 'P', the Cholesky decomposition. Symmetry can reduce the
%              computation time by about a factor of two.
%
%      - VERB: display status info if VERB = 1 (default: VERB = 0). This prints
%              to screen a table tracking compression statistics through level.
%              Each level is indexed as 'L-D', where L indicates the tree level
%              and D the current dimensionality of that level. Special levels:
%              'T', tree sorting.
%
%    Primary references:
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. Comm. Pure Appl. Math. 69 (7):
%        1314-1353, 2016.
%
%    Other references:
%
%      E. Corona, P.-G. Martinsson, D. Zorin. An O(N) direct solver for
%        integral equations on the plane. Appl. Comput. Harmon. Anal. 38 (2):
%        284-317, 2015.
%
%    See also HIFIE2X, HIFIE3, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV, HYPOCT, ID, RSKELF.

function F = hifie2(A,x,occ,rank_or_tol,pxyfun,opts)

  % set default parameters
  if nargin < 5, pxyfun = []; end
  if nargin < 6, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'ext'), opts.ext = []; end
  if ~isfield(opts,'skip'), opts.skip = 0; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
  assert(opts.skip >= 0,'FLAM:hifie2:invalidSkip', ...
         'Skip parameter must be nonnegative.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:hifie2:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')
  if strcmpi(opts.symm,'h') && isoctave()
    warning('FLAM:hifie2:octaveLDL','No LDL decomposition in Octave; using LU.')
    opts.symm = 'n';
  end

  % print header
  if opts.verb
    fprintf([repmat('-',1,71) '\n'])
    fprintf('%5s | %6s | %19s | %19s | %10s\n', ...
            'lvl','nblk','start/end npts','start/end npts/blk','time (s)')
    fprintf([repmat('-',1,71) '\n'])
  end

  % build tree
  N = size(x,2);
  ts = tic;
  t = hypoct(x,occ,opts.lvlmax,opts.ext);
  te = toc(ts);
  if opts.verb, fprintf('%5s | %63.2e\n','t',te); end

  % count nonempty boxes at each level
  pblk = zeros(t.nlvl+1,1);
  for lvl = 1:t.nlvl
    pblk(lvl+1) = pblk(lvl);
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      if ~isempty(t.nodes(i).xi), pblk(lvl+1) = pblk(lvl+1) + 1; end
    end
  end

  % initialize
  mn = t.lvp(end);  % maximum capacity for matrix factors
  e = cell(mn,1);
  F = struct('sk',e,'rd',e,'T',e,'L',e,'U',e,'p',e,'E',e,'F',e);
  F = struct('N',N,'nlvl',0,'lvp',zeros(1,2*t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = true(N,1);  % which points remain?
  M = sparse(N,N);  % sparse matrix for Schur complement updates
  nz = 128;         % initial capacity for sparse matrix workspace
  I = zeros(nz,1);
  J = zeros(nz,1);
  V = zeros(nz,1);
  P = zeros(N,1);   % for indexing

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    l = t.lrt/2^(lvl - 1);
    nbox = t.lvp(lvl+1) - t.lvp(lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).xi = [t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]];
    end

    % loop over dimensions
    for d = [2 1]
      ts = tic;

      % dimension reduction
      if d < 2
        if lvl == 1, break; end                     % done if at root
        if lvl > t.nlvl - opts.skip, continue; end  % continue if in skip stage

        % generate edge centers
        ctr = zeros(4*nbox,2);
        box2ctr = cell(nbox,1);
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)  % cells on this level
          j = i - t.lvp(lvl);              % local cell index
          idx = 4*(j-1)+1:4*j;             % edge indices
          off = [0 -1; -1  0; 0 1; 1 0];   % offset from cell center
          ctr(idx,:) = t.nodes(i).ctr + 0.5*l*off;  % edge centers
          box2ctr{j} = idx;                % mapping from each cell to its edges
        end

        % restrict to unique shared centers
        idx = round(2*(ctr - t.nodes(1).ctr)/l);  % displacement from root
        [~,i,j] = unique(idx,'rows');             % unique indices
        p = find(histc(j,1:max(j)) > 1);          % shared indices
        ctr = ctr(i(p),:);                        % remaining centers
        idx = zeros(size(idx,1),1);  % mapping from each index to ...
        idx(p) = 1:length(p);        % ... remaining index or none
        for box = 1:nbox, box2ctr{box} = nonzeros(idx(j(box2ctr{box})))'; end

        % initialize center data structure
        nb = size(ctr,1);
        e = cell(nb,1);
        blocks = struct('ctr',e,'xi',e,'prnt',e,'nbr1',e,'nbr2',e);

        % sort points by centers
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];  % points in this cell
          i = box2ctr{box};                   % associated centers
          dx = x(1,xi) - ctr(i,1);
          dy = x(2,xi) - ctr(i,2);
          dist = sqrt(dx.^2 + dy.^2);
          [~,near] = max(dist == min(dist));  % nearest center to each point
          P(xi) = box2ctr{box}(near);         % assign points to first nearest
        end
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          if isempty(xi), continue; end
          cnt = histc(P(xi),1:nb);              % num of points for each center
          [~,idx] = sort(P(xi)); xi = xi(idx);  % sort points by centers
          p = [0; cumsum(cnt)];  % index array from sorted points to centers
          for i = box2ctr{box}
            blocks(i).xi = [blocks(i).xi xi(p(i)+1:p(i+1))];
            blocks(i).prnt = [blocks(i).prnt (t.lvp(lvl)+box)*ones(1,cnt(i))];
          end
        end

        % restrict to nonempty centers
        cnt = histc(P(rem),1:nb);
        idx = cnt > 0;
        ctr = ctr(idx,:);
        blocks = blocks(idx);
        nb = length(blocks);   % number of nonempty centers
        for i = 1:nb, blocks(i).ctr = ctr(i,:); end
        p = cumsum(cnt == 0);  % how many empty boxes before this one?
        for box = 1:nbox
          box2ctr{box} = box2ctr{box}(idx(box2ctr{box}));  % nonempty only
          box2ctr{box} = box2ctr{box} - p(box2ctr{box})';  % re-index
        end

        % find neighbors for each center
        proc = false(nb,1);
        for box = 1:nbox
          slf = box2ctr{box};
          nbr = t.nodes(t.lvp(lvl)+box).nbor;  % neighbors for this cell
          nbr1 = nbr(nbr <= t.lvp(lvl));       % nbr1 = higher-level cells
          for i = slf, blocks(i).nbr1 = [blocks(i).nbr1 nbr1]; end
          nbr2 = nbr(nbr > t.lvp(lvl)) - t.lvp(lvl);  % nbr2 = same-level ...
          nbr2 = unique([slf box2ctr{nbr2}]);         %        ... centers
          dx = abs(round((ctr(slf,1) - ctr(nbr2,1)')/l))';
          dy = abs(round((ctr(slf,2) - ctr(nbr2,2)')/l))';
          near = dx <= 1 & dy <= 1;  % check if actually neighbors by geometry
          for i = 1:length(slf)
            j = slf(i);
            if proc(j), continue; end    % already processed
            k = nbr2(near(:,i));
            blocks(j).nbr2 = k(k ~= j);  % don't include self
            proc(j) = true;
          end
        end
      end

      % initialize for compression/factorization
      nlvl = nlvl + 1;
      if d == 2, nb = t.lvp(lvl+1) - t.lvp(lvl);  % use cells
      else,      nb = length(blocks);             % use centers
        for i = t.lvp(lvl)+1:t.lvp(lvl+1), t.nodes(i).xi = []; end
      end
      nrem1 = sum(rem);  % remaining points at start
      nz = 0;

      % loop over blocks
      for i = 1:nb
        if d == 2
          j = t.lvp(lvl) + i;  % global tree index
          blk = t.nodes(j);
          nbr = [t.nodes(blk.nbor).xi];
        else
          blk = blocks(i);
          nbr = [[t.nodes(blk.nbr1).xi] [blocks(blk.nbr2).xi]];
        end
        slf = blk.xi;
        nslf = length(slf);
        sslf = sort(slf);

        % compute proxy interactions and subselect neighbors
        Kpxy = zeros(0,nslf);
        if lvl > 2
          if isempty(pxyfun), nbr = setdiff(find(rem),slf);
          else, [Kpxy,nbr] = pxyfun(x,slf,nbr,l,blk.ctr);
          end
        end

        % add neighbors with modified interactions
        [nbr_mod,~] = find(M(:,slf));
        nbr_mod = unique(nbr_mod);
        nbr_mod = nbr_mod(~ismemb(nbr_mod,sslf));
        nbr = unique([nbr(:); nbr_mod(:)]);

        % compress off-diagonal block
        K1 = full(A(nbr,slf));
        if strcmpi(opts.symm,'n'), K1 = [K1; full(A(slf,nbr))']; end
        K2 = spget(M,nbr,slf);
        if strcmpi(opts.symm,'n'), K2 = [K2; spget(M,slf,nbr)']; end
        K = [K1 + K2; Kpxy];
        [sk,rd,T] = id(K,rank_or_tol,0);  % no randomization for better ranks

        % restrict to skeletons
        if d == 2
          t.nodes(j).xi = slf(sk);
        else
          for j = sk
            t.nodes(blk.prnt(j)).xi = [t.nodes(blk.prnt(j)).xi slf(j)];
          end
        end

        % move on if no compression
        if isempty(rd), continue; end
        rem(slf(rd)) = 0;

        % compute factors
        K = full(A(slf,slf)) + spget(M,slf,slf);
        if strcmpi(opts.symm,'s'), K(rd,:) = K(rd,:) - T.'*K(sk,:);
        else,                      K(rd,:) = K(rd,:) - T' *K(sk,:);
        end
        K(:,rd) = K(:,rd) - K(:,sk)*T;
        if strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s')
          [L,U,p] = lu(K(rd,rd),'vector');
          E = K(sk,rd)/U;
          G = L\K(rd(p),sk);
        elseif strcmpi(opts.symm,'h')
          [L,U,p] = ldl(K(rd,rd),'vector');
          U = diag(U);
          E = (K(sk,rd(p))/L')./U.';
          G = [];
        elseif strcmpi(opts.symm,'p')
          L = chol(K(rd,rd),'lower');
          E = K(sk,rd)/L';
          U = []; p = []; G = [];
        end

        % update self-interaction
        if     strcmpi(opts.symm,'h'), X = -E*(U.*E');
        elseif strcmpi(opts.symm,'p'), X = -E*E';
        else,                          X = -E*G;
        end
        [I_,J_] = ndgrid(slf(sk));
        [I,J,V,nz] = sppush3(I,J,V,nz,I_,J_,X);

        % store matrix factors
        n = n + 1;
        if mn < n
          e = cell(mn,1);
          s = struct('sk',e,'rd',e,'T',e,'L',e,'U',e,'p',e,'E',e,'F',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
        F.factors(n).sk = slf(sk);
        F.factors(n).rd = slf(rd);
        F.factors(n).T = T;
        F.factors(n).L = L;
        F.factors(n).U = U;
        F.factors(n).p = p;
        F.factors(n).E = E;
        F.factors(n).F = G;
      end
      F.lvp(nlvl+1) = n;

      % update modified entries
      [I_,J_,V_] = find(M);
      idx = rem(I_) & rem(J_);
      [I,J,V,nz] = sppush3(I,J,V,nz,I_(idx),J_(idx),V_(idx));
      M = sparse(I(1:nz),J(1:nz),V(1:nz),N,N);
      te = toc(ts);

      % print summary
      if opts.verb
        nrem2 = sum(rem);  % remaining points at end
        nblk = pblk(lvl) + nb;  % nonempty up to this level
        fprintf('%3d-%1d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
                lvl,d,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,te)
      end
    end
  end

  % finish
  F.nlvl = nlvl;
  F.lvp = F.lvp(1:nlvl+1);
  F.factors = F.factors(1:n);
  if opts.verb, fprintf([repmat('-',1,71) '\n']); end
end