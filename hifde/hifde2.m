% HIFDE2   Hierarchical interpolative factorization for differential equations
%          with nearest neighbor interactions on a regular mesh in 2D.
%
%    F = HIFDE2(A,N,OCC,RANK_OR_TOL) produces a factorization F of the sparse
%    interaction matrix A on the interior vertices of a regular N x N finite
%    element mesh of the unit square with leaf size OCC x OCC using local
%    skeletonization precision parameter RANK_OR_TOL.
%
%    F = HIFDE2(A,N,OCC,RANK_OR_TOL,OPTS) also passes various options to the
%    algorithm. Valid options include:
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
%    See also HIFDE2X, HIFDE3, HIFDE3X, HIFDE_CHOLMV, HIFDE_CHOLSV, HIFDE_DIAG,
%    HIFDE_LOGDET, HIFDE_MV, HIFDE_SPDIAG, HIFDE_SV, HYPOCT, ID.

function F = hifde2(A,n,occ,rank_or_tol,opts)
  start = tic;

  % set default parameters
  if nargin < 5
    opts = [];
  end
  if ~isfield(opts,'lvlmax')
    opts.lvlmax = Inf;
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
  assert(n > 0,'FLAM:hifde2:nonpositiveMeshSize','Mesh size must be positive.')
  if occ <= 0
    assert(isfinite(opts.lvlmax),'FLAM:hifde2:invalidLvlmax', ...
          'Maximum tree depth must be finite if leaf occupancy is zero.')
  end
  assert(opts.lvlmax >= 1,'FLAM:hifde2:invalidLvlmax', ...
         'Maximum tree depth must be at least 1.')
  assert(opts.skip >= 0,'FLAM:hifde2:negativeSkip', ...
         'Skip parameter must be nonnegative.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:hifde2:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')
  if strcmpi(opts.symm,'h') && isoctave()
    warning('FLAM:rskelf:octaveLDL','No LDL decomposition in Octave; using LU.')
    opts.symm = 's';
  end

  % print header
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
  end

  % initialize
  nd = n - 1;
  N = nd^2;
  nlvl = min(opts.lvlmax,ceil(max(0,log2(n/occ)))+1);
  mn = (4^nlvl - 1)/3;
  e = cell(mn,1);
  F = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',nlvl,'lvp',zeros(1,2*nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  nf = 0;
  grd = reshape(1:N,nd,nd);
  rem = true(nd,nd);
  mnz = 128;
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  S = zeros(mnz,1);
  P = zeros(N,1);

  % set initial width
  w = n;
  for lvl = 1:F.nlvl
    w = ceil(w/2);
  end

  % loop over tree levels
  for lvl = F.nlvl:-1:1
    w = 2*w;
    nb = ceil(n/w);

    % loop over dimensions
    for d = [2 1]
      tic
      nrem1 = sum(rem(:));

      % block elimination
      if d == 2
        nblk = nb^2;
        e = cell(nblk,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nblk_ = nblk;
        nblk = 0;

        % loop over cells
        for i = 1:nb
          for j = 1:nb

            % set up indices
            ia = (i - 1)*w;
            ib =  i     *w;
            is = max(1,ia):min(nd,ib);
            ja = (j - 1)*w;
            jb =  j     *w;
            js = max(1,ja):min(nd,jb);

            % initialize local arrays
            grd_ = grd(is,js);
            rem_ = rem(is,js);
            slf = grd_(rem_);
            slf = slf(:)';

            % find self indices
            idx = slf - 1;
            jj = floor(idx/nd);
            ii = idx - nd*jj;
            ii = ii + 1;
            jj = jj + 1;

            % skeletonize (eliminate interior nodes)
            in = ii > ia & ii < ib & jj > ja & jj < jb;
            sk = find(~in);
            rd = find( in);
            sk = sk(:)';
            rd = rd(:)';

            % move on if no compression
            if isempty(rd)
              continue
            end
            rem(slf(rd)) = 0;

            % store data
            nblk = nblk + 1;
            blocks(nblk).slf = slf;
            blocks(nblk).sk = sk;
            blocks(nblk).rd = rd;
          end
        end

      % skeletonization (dimension reduction)
      else

        % continue if in skip stage
        if lvl > F.nlvl - opts.skip
          continue
        end

        % initialize
        nblk = 2*nb*(nb - 1);
        e = cell(nblk,1);
        blocks = struct('slf',e,'sk',e,'rd',e,'T',e);
        nblk_ = nblk;
        nblk = 0;

        % loop over edges
        for i = 1:2*nb-1
          mi = mod(i,2);
          for j = 1:2*nb-1
            mj = mod(j,2);
            if mi + mj ~= 1
              continue
            end

            % set up indices
            ib = floor(i/2);
            jb = floor(j/2);
            if mi
              is = ib*w + (1:w-1);
              in = ib*w + (0:w  );
              js = jb*w;
              jn = jb*w + (-w:w);
            elseif mj
              is = ib*w;
              in = ib*w + (-w:w);
              js = jb*w + (1:w-1);
              jn = jb*w + (0:w  );
            end

            % restrict to domain
            is = is(is > 0 & is < n);
            in = in(in > 0 & in < n);
            js = js(js > 0 & js < n);
            jn = jn(jn > 0 & jn < n);

            % initialize local arrays
            grd_ = grd(is,js);
            rem_ = rem(is,js);
            slf = grd_(rem_);
            slf = slf(:)';
            grd_ = grd(in,jn);
            rem_ = rem(in,jn);
            nbr = grd_(rem_);
            nbr = nbr(:)';
            nbr = nbr(~ismemb(nbr,slf));

            % compute interaction matrix
            [K,P] = spget(A,nbr,slf,P);
            if strcmpi(opts.symm,'n')
              [tmp,P] = spget(A,slf,nbr,P);
              K = [K; tmp'];
            end

            % skeletonize
            [sk,rd,T] = id(K,rank_or_tol);

            % move on if no compression
            if isempty(rd)
              continue
            end
            rem(slf(rd)) = 0;

            % store data
            nblk = nblk + 1;
            blocks(nblk).slf = slf;
            blocks(nblk).sk = sk;
            blocks(nblk).rd = rd;
            blocks(nblk).T = T;
          end
        end
      end
      blocks = blocks(1:nblk);

      % initialize
      nlvl = nlvl + 1;
      nz = 0;

      % loop over stored data
      for i = 1:nblk
        slf = blocks(i).slf;
        sk = blocks(i).sk;
        rd = blocks(i).rd;
        T = blocks(i).T;

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
        nf = nf + 1;
        while mn < nf
          e = cell(mn,1);
          s = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
        F.factors(nf).sk = slf(sk);
        F.factors(nf).rd = slf(rd);
        F.factors(nf).T = T;
        F.factors(nf).E = E;
        F.factors(nf).F = G;
        F.factors(nf).L = L;
        F.factors(nf).U = U;
      end
      F.lvp(nlvl+1) = nf;

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
      nblk = nblk_;
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
  F.factors = F.factors(1:nf);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end