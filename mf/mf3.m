% MF3  Multifrontal factorization for nearest neighbor interactions on a
%      regular mesh in 3D.
%
%    F = MF3(A,N,OCC) produces a factorization F of the sparse interaction
%    matrix A on the interior vertices of a regular N x N x N finite element
%    mesh of the unit cube with leaf size OCC x OCC x OCC.
%
%    F = MF3(A,N,OCC,OPTS) also passes various options to the algorithm. Valid
%    options include:
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
%      I.S. Duff, J.K. Reid. The multifrontal solution of indefinite sparse
%        symmetric linear equations. ACM Trans. Math. Softw. 9 (3): 302-325,
%        1983.
%
%      A. George. Nested dissection of a regular finite element mesh. SIAM J.
%        Numer. Anal. 10 (2): 345-363, 1973.
%
%      B.M. Irons. A frontal solution program for finite element analysis. Int.
%        J. Numer. Meth. Eng. 2: 5-32, 1970.
%
%    See also MF2, MF_CHOLMV, MF_CHOLSV, MF_DIAG, MF_LOGDET, MF_MV, MF_SPDIAG,
%    MF_SV, MFX.

function F = mf3(A,n,occ,opts)
  start = tic;

  % set default parameters
  if nargin < 4
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
  assert(n > 0,'FLAM:mf3:nonpositiveMeshSize','Mesh size must be positive.')
  if occ <= 0
    assert(isfinite(opts.lvlmax),'FLAM:mf3:invalidLvlmax', ...
          'Maximum tree depth must be finite if leaf occupancy is zero.')
  end
  assert(opts.lvlmax >= 1,'FLAM:mf3:invalidLvlmax', ...
         'Maximum tree depth must be at least 1.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:mf3:invalidSymm', ...
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
  N = nd^3;
  nlvl = min(opts.lvlmax,ceil(max(0,log2(n/occ)))+1);
  nbox = (8^nlvl - 1)/7;
  e = cell(nbox,1);
  F = struct('sk',e,'rd',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',nlvl,'lvp',zeros(1,nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  nf = 0;
  grd = reshape(1:N,nd,nd,nd);
  rem = true(nd,nd,nd);
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
    tic
    nlvl = nlvl + 1;
    w = 2*w;
    nb = ceil(n/w);
    nrem1 = sum(rem(:));
    nz = 0;

    % loop over cells
    for i = 1:nb
      for j = 1:nb
        for k = 1:nb

          % set up indices
          ia = (i - 1)*w;
          ib =  i     *w;
          is = max(1,ia):min(nd,ib);
          ja = (j - 1)*w;
          jb =  j     *w;
          js = max(1,ja):min(nd,jb);
          ka = (k - 1)*w;
          kb =  k     *w;
          ks = max(1,ka):min(nd,kb);
          [ii,jj,kk] = ndgrid(is,js,ks);

          % initialize local arrays
          grd_ = grd(is,js,ks);
          rem_ = rem(is,js,ks);
          idx = zeros(size(grd_));
          idx(rem_) = 1:sum(rem_(:));
          slf = grd_(rem_);
          slf = slf(:)';

          % skeletonize (eliminate interior nodes)
          in = ii ~= ia & ii ~= ib & jj ~= ja & jj ~= jb & kk ~= ka & kk ~= kb;
          sk = idx(rem_ & ~in);
          rd = idx(rem_ &  in);
          sk = sk(:)';
          rd = rd(:)';

          % move on if no compression
          if isempty(rd)
            continue
          end
          rem(slf(rd)) = 0;

          % compute factors
          [K,P] = spget(A,slf,slf,P);
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
          F.factors(nf).sk = slf(sk);
          F.factors(nf).rd = slf(rd);
          F.factors(nf).E = E;
          F.factors(nf).F = G;
          F.factors(nf).L = L;
          F.factors(nf).U = U;
        end
      end
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
    if opts.verb
      nrem2 = sum(rem(:));
      nblk = nb^3;
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
              lvl,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,toc)
    end
  end

  % finish
  F.factors = F.factors(1:nf);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end