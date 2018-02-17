% MFX  Multifrontal factorization on general meshes.
%
%    F = MFX(A,X,OCC) produces a factorization F of the sparse interaction
%    matrix A on the points X using tree occupancy parameter OCC.
%
%    F = MFX(A,X,OCC,OPTS) also passes various options to the algorithm. Valid
%    options include:
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
%    See also HYPOCT, MF2, MF3, MF_CHOLMV, MF_CHOLSV, MF_DIAG, MF_LOGDET, MF_MV,
%    MF_SPDIAG, MF_SV.

function F = mfx(A,x,occ,opts)
  start = tic;

  % set default parameters
  if nargin < 4
    opts = [];
  end
  if ~isfield(opts,'lvlmax')
    opts.lvlmax = Inf;
  end
  if ~isfield(opts,'ext')
    opts.ext = [];
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
         'FLAM:mfx:invalidSymm', ...
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
  F = struct('sk',e,'rd',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',t.nlvl,'lvp',zeros(1,t.nlvl+1),'factors',F,'symm', ...
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
    tic
    nlvl = nlvl + 1;
    nrem1 = sum(rem);
    nz = 0;

    % form matrix transpose
    if strcmpi(opts.symm,'n')
      B = A';
    end

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).xi = unique([t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]]);
    end

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
      n = n + 1;
      F.factors(n).sk = slf(sk);
      F.factors(n).rd = slf(rd);
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