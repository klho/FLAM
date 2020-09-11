% HIFDE2X  Hierarchical interpolative factorization for differential equations
%          in 2D.
%
%    The hierarchical interpolative factorization is an extension of the
%    standard multifrontal factorization with additional skeletonization to
%    achieve improved complexities in 2D and above. Specifically, in 2D, block
%    elimination reduces cells to edges as in MFX, then edge skeletonization
%    further reduces edges to points, assuming that the interactions between
%    different edges are low-rank. The output is an approximation of a sparse
%    matrix A as a generalized LDU (or LDL/Cholesky) decomposition.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N).
%
%    See HIFDE2 for an optimized version specialized for nearest-neighbor
%    interactions on a regular mesh.
%
%    F = HIFDE2X(A,X,OCC,RANK_OR_TOL) produces a factorization F of the matrix A
%    acting on the points X using tree occupancy parameter OCC and local
%    precision parameter RANK_OR_TOL. See HYPOCT and ID for details.
%
%    F = HIFDE2X(A,X,OCC,RANK_OR_TOL,OPTS) also passes various options to the
%    algorithm. Valid options include:
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = INF). See HYPOCT.
%
%      - EXT: set the root node extent to [EXT(D,1) EXT(D,2)] along dimension D.
%             If EXT is empty (default), then the root extent is calculated from
%             the data. See HYPOCT.
%
%      - TMAX: ID interpolation matrix entry bound (default: TMAX = 2). See ID.
%
%      - RRQR_ITER: maximum number of RRQR refinement iterations in ID (default:
%                   RRQR_ITER = INF). See ID.
%
%      - SKIP: skip the additional dimension reductions on the first SKIP levels
%              (default: SKIP = 0). More generally, this can be a logical
%              function of the form SKIP(LVL,L) that specifies whether to skip a
%              particular reduction based on the current tree level LVL above
%              the bottom and node size L.
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
%        operators: differential equations. Comm. Pure Appl. Math. 69 (8):
%        1415-1451, 2016.
%
%    Other references:
%
%      A. Gillman, P.G. Martinsson. A direct solver with O(N) complexity for
%        variable coefficient elliptic PDEs discretized via a high-order
%        composite spectral collocation method. SIAM J. Sci. Comput. 36 (4):
%        A2023-A2046, 2014.
%
%      A. Gillman, P.G. Martinsson. An O(N) algorithm for constructing the
%        solution operator to 2D elliptic boundary value problems in the absence
%        of body loads. Adv. Comput. Math. 40 (4): 773-796, 2014.
%
%      J. Xia. Randomized sparse direct solvers. SIAM J. Matrix Anal. Appl. 34
%        (1): 197-227, 2013.
%
%      J. Xia, S. Chandrasekaran, M. Gu, X.S. Li. Superfast multifrontal method
%        for large structured linear systems of equations. SIAM J. Matrix Anal.
%        Appl. 31 (3): 1382-1411, 2009.
%
%    See also HIFDE2, HIFDE3, HIFDE3X, HIFDE_CHOLMV, HIFDE_CHOLSV, HIFDE_DIAG,
%    HIFDE_LOGDET, HIFDE_MV, HIFDE_SPDIAG, HIFDE_SV, HYPOCT, ID.

function F = hifde2x(A,x,occ,rank_or_tol,opts)

  % set default parameters
  if nargin < 5, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'ext'), opts.ext = []; end
  if ~isfield(opts,'Tmax'), opts.Tmax = 2; end
  if ~isfield(opts,'rrqr_iter'), opts.rrqr_iter = Inf; end
  if ~isfield(opts,'skip'), opts.skip = 0; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
  if isnumeric(opts.skip), opts.skip = @(lvl,l)(lvl < opts.skip); end
  opts.symm = chksymm(opts.symm);
  if opts.symm == 'h' && isoctave()
    warning('FLAM:hifde2x:octaveLDL','No LDL decomposition in Octave; using LU.')
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
  nz = 128;         % initial capacity for sparse matrix updates
  I = zeros(nz,1);
  J = zeros(nz,1);
  V = zeros(nz,1);
  P = zeros(N,1);   % for indexing

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    l = t.l(:,lvl);
    nbox = t.lvp(lvl+1) - t.lvp(lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).xi = unique([t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]]);
    end

    % loop over dimensions
    for d = [2 1]
      ts = tic;
      nrem1 = nnz(rem);  % remaining points at start

      % form matrix transpose for fast row access
      if opts.symm == 'n', Ac = A'; end

      % block elimination
      if d == 2
        nblk = nbox;
        e = cell(nblk,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nb = 0;

        % loop over nodes
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          slf = t.nodes(i).xi;
          sslf = sort(slf);

          % keep only points with exterior interactions
          [I_,J_] = find(A(:,slf));
          idx = ~ismemb(I_,sslf);
          I_ = I_(idx);
          J_ = J_(idx);
          if opts.symm == 'n'
            [Ic,Jc] = find(Ac(:,slf));
            idx = ~ismemb(Ic,sslf);
            Ic = Ic(idx);
            Jc = Jc(idx);
            I_ = [I_(:); Ic(:)];
            J_ = [J_(:); Jc(:)];
            [J_,idx] = sort(J_);
            I_ = I_(idx);
          end
          sk = unique(J_)';

          % optimize by sharing skeletons among neighbors (thin separators)
          nbr = t.nodes(i).nbor;
          nbr = nbr(nbr < i);  % restrict to already processed neighbors
          if ~isempty(nbr)
            nbr = sort([t.nodes(nbr).xi]);
            isnbr = ismemb(I_,nbr);
            p = [0; find(diff(J_)); length(J_)];  % indexing array

            % remove those made redundant by neighbor skeletons
            nsk = length(sk);
            keep = true(nsk,1);
            nbrsk = [];  % neighbor skeletons to share
            for j = 1:nsk
              if ~all(isnbr(p(j)+1:p(j+1))), continue; end  % redundant if
              keep(j) = false;                     % ... interacts only with
              nbrsk = [nbrsk I_(p(j)+1:p(j+1))'];  % ... neighbor skeletons
            end
            nbrsk = unique(nbrsk);
            % prune self-skeletons and add neighbor-skeletons
            sk = [sk(keep) length(slf)+(1:length(nbrsk))];
            slf = [slf nbrsk];
          end

          % restrict to skeletons for next level
          t.nodes(i).xi = slf(sk);  % note: can expand due to neighbor sharing
          rd = find(~ismemb(1:length(slf),sort(sk)));

          % move on if no compression
          if isempty(rd), continue; end
          rem(slf(rd)) = false;

          % store data
          nb = nb + 1;
          blocks(nb).slf = slf;
          blocks(nb).sk = sk;
          blocks(nb).rd = rd;
        end

      % skeletonization
      else
        if lvl == 1, break; end                    % done if at root
        if opts.skip(t.nlvl-lvl,l), continue; end  % continue if in skip stage

        % generate edge centers
        ctr = zeros(2,4*nbox);
        box2ctr = cell(nbox,1);
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)  % cells on this level
          j = i - t.lvp(lvl);              % local cell index
          idx = 4*(j-1)+1:4*j;             % edge indices
          off = [0 -1; -1  0; 0 1; 1 0]';  % offset from cell center
          ctr(:,idx) = t.nodes(i).ctr + 0.5*l.*off;  % edge centers
          box2ctr{j} = idx;                % mapping from each cell to its edges
        end

        % restrict to unique shared centers
        idx = round(2*(ctr - t.nodes(1).ctr)./l);  % displacement from root
        [~,i,j] = unique(idx','rows');             % unique indices
        p = find(histc(j,1:max(j)) > 1);           % shared indices
        ctr = ctr(:,i(p));                         % remaining centers
        idx = zeros(size(idx,2),1);  % mapping from each index to ...
        idx(p) = 1:length(p);        % ... remaining index or none
        for box = 1:nbox, box2ctr{box} = nonzeros(idx(j(box2ctr{box})))'; end

        % initialize center data structure
        nb = size(ctr,2);
        e = cell(nb,1);
        blk = struct('ctr',e,'xi',e,'prnt',e);

        % sort points by centers
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];  % points in this cell
          i = box2ctr{box};                   % associated centers
          dx = x(1,xi) - ctr(1,i)';
          dy = x(2,xi) - ctr(2,i)';
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
            blk(i).xi = [blk(i).xi xi(p(i)+1:p(i+1))];
            blk(i).prnt = [blk(i).prnt (t.lvp(lvl)+box)*ones(1,cnt(i))];
          end
        end

        % restrict to nonempty centers
        cnt = histc(P(rem),1:nb);
        idx = cnt > 0;
        ctr = ctr(:,idx);
        blk = blk(idx);
        nb = length(blk);   % number of nonempty centers
        p = cumsum(cnt == 0);  % how many empty boxes before this one?
        for box = 1:nbox
          box2ctr{box} = box2ctr{box}(idx(box2ctr{box}));  % nonempty only
          box2ctr{box} = box2ctr{box} - p(box2ctr{box})';  % re-index
        end

        % remove duplicate points
        for i = 1:nb
          blk(i).ctr = ctr(:,i);
          [blk(i).xi,idx] = unique(blk(i).xi,'first');
          blk(i).prnt = blk(i).prnt(idx);
        end

        % initialize for compression
        nblk = nb;
        e = cell(nblk,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nb = 0;

        % clear current level
        for i = t.lvp(lvl)+1:t.lvp(lvl+1), t.nodes(i).xi = []; end

        % loop over blocks
        for i = 1:nblk
          slf = blk(i).xi;

          % find neighbors
          [nbr,~] = find(A(:,slf));
          if opts.symm == 'n'
            [nbrc,~] = find(Ac(:,slf));
            nbr = [nbr(:); nbrc(:)]';
          end
          nbr = unique(nbr);
          nbr = nbr(~ismemb(nbr,sort(slf)));

          % compress off-diagonal block
          K = spget(A,nbr,slf);
          if opts.symm == 'n', K = [K; spget(A,slf,nbr)']; end
          [sk,rd,T] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);

          % restrict to skeletons for next level
          for j = sk
            t.nodes(blk(i).prnt(j)).xi = [t.nodes(blk(i).prnt(j)).xi slf(j)];
          end

          % move on if no compression
          if isempty(rd), continue; end
          rem(slf(rd)) = false;

          % store data
          nb = nb + 1;
          blocks(nb).slf = slf;
          blocks(nb).sk = sk;
          blocks(nb).rd = rd;
          blocks(nb).T = T;
        end
      end

      % initialize for factorization
      nlvl = nlvl + 1;
      nz = 0;

      % loop over stored data
      for i = 1:nb
        slf = blocks(i).slf;
        sk = blocks(i).sk;
        rd = blocks(i).rd;
        T = blocks(i).T;

        % compute factors
        K = spget(A,slf,slf);
        if ~isempty(T)
          if opts.symm == 's', K(rd,:) = K(rd,:) - T.'*K(sk,:);
          else,                K(rd,:) = K(rd,:) - T' *K(sk,:);
          end
          K(:,rd) = K(:,rd) - K(:,sk)*T;
        end
        if opts.symm == 'n' || opts.symm == 's'
          [L,U,p] = lu(K(rd,rd),'vector');
          E = K(sk,rd)/U;
          G = L\K(rd(p),sk);
        elseif opts.symm == 'h'
          [L,U,p] = ldl(K(rd,rd),'vector');
          rd = rd(p);
          if ~isempty(T), T = T(:,p); end
          U = sparse(U);
          E = (K(sk,rd)/L')/U.';
          p = []; G = [];
        elseif opts.symm == 'p'
          L = chol(K(rd,rd),'lower');
          E = K(sk,rd)/L';
          U = []; p = []; G = [];
        end

        % update self-interaction
        if     opts.symm == 'h', X = -E*(U*E');
        elseif opts.symm == 'p', X = -E*E';
        else,                    X = -E*G;
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
      [I_,J_,V_] = find(A);
      idx = rem(I_) & rem(J_);
      [I,J,V,nz] = sppush3(I,J,V,nz,I_(idx),J_(idx),V_(idx));
      A = sparse(I(1:nz),J(1:nz),V(1:nz),N,N);
      te = toc(ts);

      % print summary
      if opts.verb
        nrem2 = nnz(rem);         % remaining points at end
        nblk = pblk(lvl) + nblk;  % nonempty up to this level
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