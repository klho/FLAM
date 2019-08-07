% MF2  Multifrontal factorization for nearest-neighbor interactions on a
%      regular mesh in 2D.
%
%    This is an optimization of MFX for the special case of nearest-neighbor
%    interactions on a regular mesh in 2D. A given mesh node is allowed to
%    interact only with itself and its eight immediate neighbors, thus yielding
%    width-one separators in the multifrontal tree. Matrix indices are assigned
%    to the nodes according to the natural ordering.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N^(3/2)).
%
%    F = MF2(A,N,OCC) produces a factorization F of the matrix A acting on the
%    nodes of a regular (N-1) x (N-1) mesh of the unit square with leaf size
%    (OCC-1) x (OCC-1).
%
%    F = MF2(A,N,OCC,OPTS) also passes various options to the algorithm. Valid
%    options include:
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
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
%    See also MF3, MF_CHOLMV, MF_CHOLSV, MF_DIAG, MF_LOGDET, MF_MV, MF_SPDIAG,
%    MF_SV, MFX.

function F = mf2(A,n,occ,opts)

  % set default parameters
  if nargin < 4, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
  assert(n > 0,'FLAM:mf2:invalidMeshSize','Mesh size must be positive.')
  assert(occ > 0,'FLAM:mf2:invalidOcc','Leaf occupancy must be positive.')
  assert(opts.lvlmax >= 1,'FLAM:mf2:invalidLvlmax', ...
         'Maximum tree depth must be at least 1.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:mf2:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')
  if strcmpi(opts.symm,'h') && isoctave()
    warning('FLAM:mf2:octaveLDL','No LDL decomposition in Octave; using LU.')
    opts.symm = 'n';
  end

  % print header
  if opts.verb
    fprintf([repmat('-',1,69) '\n'])
    fprintf('%3s | %6s | %19s | %19s | %10s\n', ...
            'lvl','nblk','start/end npts','start/end npts/blk','time (s)')
    fprintf([repmat('-',1,69) '\n'])
  end

  % initialize
  nd = n - 1;  % number of mesh nodes in one dimension
  N = nd^2;    % total number of nodes
  nlvl = min(opts.lvlmax,ceil(max(0,log2(n/occ)))+1);  % number of tree levels
  nbox = (4^nlvl - 1)/3;  % number of factors in multifrontal tree
  e = cell(nbox,1);
  F = struct('sk',e,'rd',e,'L',e,'U',e,'p',e,'E',e,'F',e);
  F = struct('N',N,'nlvl',nlvl,'lvp',zeros(1,nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  nf = 0;
  grd = reshape(1:N,nd,nd);  % index mapping to each node
  rem = true(nd,nd);         % which nodes remain?
  mnz = 128;                 % maximum capacity for sparse matrix updates
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  S = zeros(mnz,1);

  % set initial width
  w = n;
  for lvl = 1:F.nlvl, w = ceil(w/2); end

  % loop over tree levels
  for lvl = F.nlvl:-1:1
    ts = tic;
    nlvl = nlvl + 1;
    w = 2*w;              % cell width
    nb = ceil(n/w);       % number of cells in each dimension
    nrem1 = sum(rem(:));  % remaining nodes at start
    nz = 0;

    % loop over cells
    for i = 1:nb, for j = 1:nb

      % set up indices
      ia = (i - 1)*w;
      ib =  i     *w;
      is = max(1,ia):min(nd,ib);
      ja = (j - 1)*w;
      jb =  j     *w;
      js = max(1,ja):min(nd,jb);
      [ii,jj] = ndgrid(is,js);

      % initialize local arrays
      grd_ = grd(is,js);
      rem_ = rem(is,js);
      idx = zeros(size(grd_));
      idx(rem_) = 1:sum(rem_(:));
      slf = grd_(rem_);
      slf = slf(:)';

      % skeletonize, i.e., eliminate interior nodes
      in = ii ~= ia & ii ~= ib & jj ~= ja & jj ~= jb;
      sk = idx(rem_ & ~in);
      rd = idx(rem_ &  in);
      sk = sk(:)';
      rd = rd(:)';

      % move on if no compression
      if isempty(rd), continue; end
      rem(slf(rd)) = 0;

      % compute factors
      K = spget(A,slf,slf);
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
      if     strcmpi(opts.symm,'h'), S_ = -E*(U.*E');
      elseif strcmpi(opts.symm,'p'), S_ = -E*E';
      else,                          S_ = -E*G;
      end
      [I_,J_] = ndgrid(slf(sk));
      m = length(sk)^2;
      nz_new = nz + m;
      if mnz < nz_new
        while mnz < nz_new, mnz = 2*mnz; end
        e = zeros(mnz-length(I),1);
        I = [I; e];
        J = [J; e];
        S = [S; e];
      end
      I(nz+1:nz+m) = I_(:);
      J(nz+1:nz+m) = J_(:);
      S(nz+1:nz+m) = S_(:);
      nz = nz + m;

      % store matrix factors
      nf = nf + 1;
      F.factors(nf).sk = slf(sk);
      F.factors(nf).rd = slf(rd);
      F.factors(nf).L = L;
      F.factors(nf).U = U;
      F.factors(nf).p = p;
      F.factors(nf).E = E;
      F.factors(nf).F = G;
    end, end
    F.lvp(nlvl+1) = nf;

    % update sparse matrix
    [I_,J_,S_] = find(A);     % pull existing entries
    idx = rem(I_) & rem(J_);  % keep only those needed for next level
    I_ = I_(idx);
    J_ = J_(idx);
    S_ = S_(idx);
    m = length(S_);
    nz_new = mnz + m;
    if mnz < nz_new
      while mnz < nz_new, mnz = 2*mnz; end
      e = zeros(mnz-length(I),1);
      I = [I; e];
      J = [J; e];
      S = [S; e];
    end
    I(nz+1:nz+m) = I_;        % apply on top of queued updates
    J(nz+1:nz+m) = J_;
    S(nz+1:nz+m) = S_;
    nz = nz + m;
    A = sparse(I(1:nz),J(1:nz),S(1:nz),N,N);
    te = toc(ts);

    % print summary
    if opts.verb
      nrem2 = sum(rem(:));  % remaining nodes at end
      nblk = nb^2;
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
              lvl,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,te)
    end
  end

  % finish
  F.factors = F.factors(1:nf);
  if opts.verb, fprintf([repmat('-',1,69) '\n']); end
end