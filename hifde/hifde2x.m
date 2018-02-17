% HIFDE2X  Hierarchical interpolative factorization for differential equations
%          on general meshes in 2D.
%
%    F = HIFDE2X(A,X,OCC,RANK_OR_TOL) produces a factorization F of the sparse
%    interaction matrix A on the points X using tree occupancy parameter OCC and
%    local skeletonization precision parameter RANK_OR_TOL.
%
%    F = HIFDE2X(A,X,OCC,RANK_OR_TOL,OPTS) also passes various options to the
%    algorithm. Valid options include:
%
%      - EXT: set the root node extent to [EXT(I,1) EXT(I,2)] along dimension I.
%             If EXT is empty (default), then the root extent is calculated from
%             the data.
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - SKIP: skip the dimension reductions on the first SKIP levels (default:
%              SKIP = 0).
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
%      A. Gillman, P.G. Martinsson. A direct solver with O(N) complexity for
%        variable coefficient elliptic PDEs discretized via a high-order
%        composite spectral collocation method. SIAM J. Sci. Comput. 36 (4):
%        A2023-A2046, 2014.
%
%      A. Gillman, P.G. Martinsson. An O(N) algorithm for constructing the
%        solution operator to 2D elliptic boundary value problems in the absence
%        of body loads. Adv. Comput. Math. 40 (4): 773-796, 2014.
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: differential equations. Comm. Pure Appl. Math. 69 (8):
%        1415-1451, 2016.
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
  start = tic;

  % set default parameters
  if nargin < 5
    opts = [];
  end
  if ~isfield(opts,'lvlmax')
    opts.lvlmax = Inf;
  end
  if ~isfield(opts,'ext')
    opts.ext = [];
  end
  if ~isfield(opts,'skip')
    opts.skip = 0;
  end
  if ~isfield(opts,'symm')
    opts.symm = 'n';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(opts.skip >= 0,'FLAM:hifde2x:negativeSkip', ...
         'Skip parameter must be nonnegative.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:hifde2x:invalidSymm', ...
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
    fprintf(' %3s  | %63.2e (s)\n','-',toc)
  end

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

  % initialize
  mn = t.lvp(end);
  e = cell(mn,1);
  F = struct('sk',e,'rd',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',0,'lvp',zeros(1,2*t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = true(N,1);
  mnz = 128;
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  S = zeros(mnz,1);
  P = zeros(N,1);

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    l = t.lrt/2^(lvl - 1);
    nbox = t.lvp(lvl+1) - t.lvp(lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).xi = unique([t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]]);
    end

    % loop over dimensions
    for d = [2 1]
      tic
      nrem1 = sum(rem);

      % form matrix transpose
      if strcmpi(opts.symm,'n')
        B = A';
      end

      % block elimination
      if d == 2
        nb = t.lvp(lvl+1) - t.lvp(lvl);
        e = cell(nb,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nblk = nb;
        nb = 0;

        % loop over nodes
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          slf = t.nodes(i).xi;
          sslf = sort(slf);

          % skeletonize (eliminate interior nodes)
          [I_,J_] = find(A(:,slf));
          idx = ~ismemb(I_,sslf);
          I_ = I_(idx);
          J_ = J_(idx);
          if strcmpi(opts.symm,'n')
            [I__,J__] = find(B(:,slf));
            idx = ~ismemb(I__,sslf);
            I__ = I__(idx);
            J__ = J__(idx);
            I_ = [I_(:); I__(:)];
            J_ = [J_(:); J__(:)];
            [J_,idx] = sort(J_);
            I_ = I_(idx);
          end
          sk = unique(J_)';

          % optimize by sharing skeletons among neighbors
          nbr = t.nodes(i).nbor;
          nbr = nbr(nbr < i);
          if ~isempty(nbr)
            nbr = sort([t.nodes(nbr).xi]);
            idx = ismemb(J_,sk);
            I_ = I_(idx);
            J_ = J_(idx);
            nsk = length(sk);
            P(sk) = 1:nsk;
            J_ = P(J_);
            p = cumsum(histc(J_,1:nsk));
            p = [0; p(:)];
            idx = ismemb(I_,nbr);

            % remove those made redundant by neighbor skeletons
            keep = true(nsk,1);
            mod = [];
            for j = 1:nsk
              if all(idx(p(j)+1:p(j+1)))
                keep(j) = false;
                mod = [mod I_(p(j)+1:p(j+1))'];
              end
            end
            mod = unique(mod);
            sk = [sk(keep) length(slf)+(1:length(mod))];
            slf = [slf mod];
          end

          % restrict to skeletons
          t.nodes(i).xi = slf(sk);
          rd = find(~ismemb(1:length(slf),sort(sk)));

          % move on if no compression
          if isempty(rd)
            continue
          end
          rem(slf(rd)) = 0;

          % store data
          nb = nb + 1;
          blocks(nb).slf = slf;
          blocks(nb).sk = sk;
          blocks(nb).rd = rd;
        end

      % skeletonization (dimension reduction)
      else

        % continue if in skip stage
        if lvl > t.nlvl - opts.skip
          continue
        end

        % generate edge centers
        ctr = zeros(4*nbox,2);
        box2ctr = cell(nbox,1);
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          j = i - t.lvp(lvl);
          idx = 4*(j-1)+1:4*j;
          off = [0 -1; -1  0; 0 1; 1 0];
          ctr(idx,:) = bsxfun(@plus,t.nodes(i).ctr,0.5*l*off);
          box2ctr{j} = idx;
        end

        % find unique shared centers
        idx = bsxfun(@minus,ctr,t.nodes(1).ctr);
        idx = round(2*idx/l);
        [~,i,j] = unique(idx,'rows');
        idx(:) = 0;
        p = find(histc(j,1:max(j)) > 1);
        i = i(p);
        idx(p) = 1:length(p);
        ctr = ctr(i,:);
        for box = 1:nbox
          box2ctr{box} = nonzeros(idx(j(box2ctr{box})))';
        end

        % initialize
        nb = size(ctr,1);
        e = cell(nb,1);
        blk = struct('ctr',e,'xi',e,'prnt',e);

        % sort points by centers
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          i = box2ctr{box};
          dx = bsxfun(@minus,x(1,xi),ctr(i,1));
          dy = bsxfun(@minus,x(2,xi),ctr(i,2));
          dist = sqrt(dx.^2 + dy.^2);
          near = bsxfun(@eq,dist,min(dist,[],1));
          for i = 1:length(xi)
            P(xi(i)) = box2ctr{box}(find(near(:,i),1));
          end
        end
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          if ~isempty(xi)
            m = histc(P(xi),1:nb);
            p = cumsum(m);
            p = [0; p(:)];
            [~,idx] = sort(P(xi));
            xi = xi(idx);
            for j = box2ctr{box}
              blk(j).xi = [blk(j).xi xi(p(j)+1:p(j+1))];
              blk(j).prnt = [blk(j).prnt (t.lvp(lvl)+box)*ones(1,m(j))];
            end
          end
        end

        % keep only nonempty centers
        m = histc(P(rem),1:nb);
        idx = m > 0;
        ctr = ctr(idx,:);
        blk = blk(idx);
        nb = length(blk);
        p = cumsum(m == 0);
        for box = 1:nbox
          box2ctr{box} = box2ctr{box}(idx(box2ctr{box}));
          box2ctr{box} = box2ctr{box} - p(box2ctr{box})';
        end

        % remove duplicate points
        for i = 1:nb
          blk(i).ctr = ctr(i,:);
          [blk(i).xi,idx] = unique(blk(i).xi,'first');
          blk(i).prnt = blk(i).prnt(idx);
        end

        % initialize storage
        e = cell(nb,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nblk = nb;
        nb = 0;

        % clear current level
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          t.nodes(i).xi = [];
        end

        % loop over centers
        for i = 1:nblk
          slf = blk(i).xi;
          sslf = sort(slf);

          % find neighbors
          [nbr,~] = find(A(:,slf));
          nbr = unique(nbr);
          idx = ~ismemb(nbr,sslf);
          nbr = nbr(idx);
          if strcmpi(opts.symm,'n')
            [nbr_,~] = find(B(:,slf));
            idx = ~ismemb(nbr_,sslf);
            nbr_ = nbr_(idx);
            nbr = [nbr(:); nbr_(:)]';
          end
          nnbr = length(nbr);
          snbr = sort(nbr);

          % compute interaction matrix
          [K,P] = spget(A,nbr,slf,P);
          if strcmpi(opts.symm,'n')
            [tmp,P] = spget(A,slf,nbr,P);
            K = [K; tmp'];
          end

          % skeletonize
          [sk,rd,T] = id(K,rank_or_tol);

          % restrict to skeletons
          for j = sk
            t.nodes(blk(i).prnt(j)).xi = [t.nodes(blk(i).prnt(j)).xi slf(j)];
          end

          % move on if no compression
          if isempty(rd)
            continue
          end
          rem(slf(rd)) = 0;

          % store data
          nb = nb + 1;
          blocks(nb).slf = slf;
          blocks(nb).sk = sk;
          blocks(nb).rd = rd;
          blocks(nb).T = T;
        end
      end

      % initialize
      nlvl = nlvl + 1;
      nblk = pblk(lvl) + nblk;
      nz = 0;

      % loop over stored data
      for i = 1:nb
        slf = blocks(i).slf;
        sk = blocks(i).sk;
        rd = blocks(i).rd;
        T = blocks(i).T;
        sslf = sort(slf);

        % compute factors
        [K,P] = spget(A,slf,slf,P);
        if ~isempty(T)
          if strcmpi(opts.symm,'s')
            K(rd,:) = K(rd,:) - T.'*K(sk,:);
          else
            K(rd,:) = K(rd,:) - T'*K(sk,:);
          end
          K(:,rd) = K(:,rd) - K(:,sk)*T;
        end
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
          S_ = -E*G;
        elseif strcmpi(opts.symm,'h')
          S_ = -E*U*E';
        elseif strcmpi(opts.symm,'p')
          S_ = -E*E';
        end
        [I_,J_] = ndgrid(slf(sk));
        m = length(sk)^2;
        while mnz < nz + m
          e = zeros(mnz,1);
          I = [I; e];
          J = [J; e];
          S = [S; e];
          mnz = 2*mnz;
        end
        I(nz+1:nz+m) = I_(:);
        J(nz+1:nz+m) = J_(:);
        S(nz+1:nz+m) = S_(:);
        nz = nz + m;

        % store matrix factors
        n = n + 1;
        while mn < n
          e = cell(mn,1);
          s = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
        F.factors(n).sk = slf(sk);
        F.factors(n).rd = slf(rd);
        F.factors(n).T = T;
        F.factors(n).E = E;
        F.factors(n).F = G;
        F.factors(n).L = L;
        F.factors(n).U = U;
      end
      F.lvp(nlvl+1) = n;

      % update modified entries
      [I_,J_,S_] = find(A);
      idx = rem(I_) & rem(J_);
      I_ = I_(idx);
      J_ = J_(idx);
      S_ = S_(idx);
      m = length(S_);
      while mnz < nz + m
        e = zeros(mnz,1);
        I = [I; e];
        J = [J; e];
        S = [S; e];
        mnz = 2*mnz;
      end
      I(nz+1:nz+m) = I_;
      J(nz+1:nz+m) = J_;
      S(nz+1:nz+m) = S_;
      nz = nz + m;
      A = sparse(I(1:nz),J(1:nz),S(1:nz),N,N);

      % print summary
      if opts.verb
        nrem2 = sum(rem(:));
        fprintf('%3d-%1d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
                lvl,d,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,toc)
      end
      if nblk == 1
        break
      end
    end
  end

  % finish
  F.nlvl = nlvl;
  F.lvp = F.lvp(1:nlvl+1);
  F.factors = F.factors(1:n);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end