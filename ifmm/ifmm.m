% IFMM  Interpolative fast multipole method.
%
%    This is an implementation of a kernel-independent fast multipole method
%    based on the interpolative decomposition for hierarchical low-rank
%    compression. The algorithmic framework is fully algebraic and supports
%    multiplication with both the matrix and its transpose/adjoint. Extra
%    functionality over standard FMMs include optional near-field compression.
%
%    Each row/column of A is associated with a point, with the induced point
%    geometry exposing the required rank structure. In typical operation, only
%    the near-field (near-diagonal blocks) is explicitly evaluated; the far-
%    field is captured by a user-supplied "proxy" function. To avoid excessive
%    storage, the matrix should be given as a function handle implementing the
%    usual submatrix access interface.
%
%    Typical complexity for [M,N] = SIZE(A): O(M + N) in all dimensions.
%
%    F = IFMM(A,RX,CX,OCC,RANK_OR_TOL) produces a compressed representation F of
%    the matrix A acting on the row and column points RX and CX, respectively,
%    using tree occupancy parameter OCC and local precision parameter
%    RANK_OR_TOL. See HYPOCT and ID for details. Note that both row/column
%    points are sorted together in the same tree, so OCC should be set roughly
%    twice the desired leaf size. Since no proxy function is supplied, this
%    simply performs a naive compression of all far-field blocks.
%
%    F = IFMM(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) accelerates the compression using
%    the proxy function PXYFUN to capture the far field. This is a function of
%    the form
%
%      [KPXY,NBR] = PXYFUN(RC,RX,CX,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor point indices (can be modified)
%      - RC:   flag to specify row or column compression ('R' or 'C')
%      - RX:   input row points
%      - CX:   input column points
%      - SLF:  block point indices
%      - L:    block node size
%      - CTR:  block node center
%
%    The relevant arguments will be passed in by the algorithm; the user is
%    responsible for handling them. The output NBR is requested only when used
%    for near-field compression (see below); it is ignored for the far field.
%    See the examples for further details. If PXYFUN is not provided or empty
%    (default), then the code uses the naive global compression scheme.
%
%    F = IFMM(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options
%    to the algorithm. Valid options include:
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
%      - NEAR: additionally compress near field if NEAR = 1 (default: NEAR = 0).
%
%      - STORE: store no interactions if STORE = 'N' (i.e., generate them on the
%               fly in IFMM_MV), only self-interactions if STORE = 'S', only
%               near-field (including self-) interactions if STORE = 'R', and
%               all interactions if STORE = 'A' (default: STORE = 'N').
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N'). Symmetry
%              can reduce the computation time by about a factor of two.
%
%      - VERB: display status info if VERB = 1 (default: VERB = 0). This prints
%              to screen a table tracking row/column compression statistics
%              through level. Special levels: 'T', tree sorting; 'N', near-field
%              compression.
%
%    Related references:
%
%      J. Carrier, L. Greengard, V. Rokhlin. A fast adaptive multipole algorithm
%        for particle simulations. SIAM J. Sci. Stat. Comput. 9 (4): 669-686,
%        1998.
%
%      P.G. Martinsson, V. Rokhlin. An accelerated kernel-independent fast
%        multipole method in one dimension. SIAM J. Sci. Comput. 29 (3):
%        1160-1178, 2007.
%
%    See also HYPOCT, ID, IFMM_MV.

function F = ifmm(A,rx,cx,occ,rank_or_tol,pxyfun,opts)

  % set default parameters
  if nargin < 6, pxyfun = []; end
  if nargin < 7, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'ext'), opts.ext = []; end
  if ~isfield(opts,'Tmax'), opts.Tmax = 2; end
  if ~isfield(opts,'rrqr_iter'), opts.rrqr_iter = Inf; end
  if ~isfield(opts,'near'), opts.near = 0; end
  if ~isfield(opts,'store'), opts.store = 'n'; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
  opts.store = lower(opts.store);
  assert(opts.store == 'n' || opts.store == 's' || opts.store == 'r' || ...
         opts.store == 'a','FLAM:ifmm:invalidStore', ...
         'Storage parameter must be one of ''N'', ''S'', ''R'', or ''A''.')
  opts.symm = chksymm(opts.symm);
  if opts.symm == 'p', opts.symm = 'h'; end

  % print header
  if opts.verb
    fprintf([repmat('-',1,69) '\n'])
    fprintf('%3s | %6s | %19s | %19s | %10s\n', ...
            'lvl','nblk','start/end npts','start/end npts/blk','time (s)')
    fprintf([repmat('-',1,69) '\n'])
  end

  % build tree
  M = size(rx,2);
  N = size(cx,2);
  ts = tic;
  t = hypoct([rx cx],occ,opts.lvlmax,opts.ext);  % bundle row/col points
  te = toc(ts);
  P = hypoct_perm(t);                   % find bundled permutation
  idx = P > M;
  if opts.symm == 'n', Q = P(idx) - M;  % extract col permutation
  else,                Q = [];
  end
  P = P(~idx);                          % extract row permutation
  for i = 1:t.lvp(t.nlvl+1)
    xi = t.nodes(i).xi;
    idx = xi <= M;
    t.nodes(i).rxi = xi( idx);      % extract row indices
    t.nodes(i).cxi = xi(~idx) - M;  % extract col indices
    t.nodes(i).xi = [];
  end
  if opts.verb, fprintf('%3s | %63.2e\n','t',te); end

  % count nonempty boxes at each level
  pblk = zeros(t.nlvl+1,1);
  for lvl = 1:t.nlvl
    pblk(lvl+1) = pblk(lvl);
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      if isempty([t.nodes(i).rxi t.nodes(i).cxi]), continue; end
      pblk(lvl+1) = pblk(lvl+1) + 1;
    end
  end

  % find direct interactions
  for i = 1:t.lvp(end), t.nodes(i).dir = []; end
  for lvl = 1:t.nlvl
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      if isempty([t.nodes(i).rxi t.nodes(i).cxi]), continue; end  % skip empty
      for j = t.nodes(i).nbor  % include nonempty neighbors
        if isempty([t.nodes(j).rxi t.nodes(j).cxi]), continue; end
        t.nodes(i).dir = [t.nodes(i).dir j];
        % neighbor relation is bijective: keepy only one of (i,j) and (j,i)
        if j <= t.lvp(lvl), t.nodes(j).dir = [t.nodes(j).dir i]; end
      end
    end
  end

  % initialize
  nlvl = t.nlvl + 1;  % extra "level" for near field
  nbox = t.lvp(end);
  e = cell(nbox,1);
  % submatrix entries: 's', self; 'e', external; 'i', incoming; 'o', outgoing;
  %                    'D' for diagonal interactions and 'B' for others
  B = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
  U = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);  % ID matrices
  F = struct('M',M,'N',N,'P',P,'Q',Q,'nlvl',nlvl,'lvpb',zeros(1,nlvl+2), ...
             'lvpu',zeros(1,nlvl+1),'B',B,'U',U,'store',opts.store,'symm', ...
             opts.symm);
  nb = 0; mnb = nbox;  % number of B nodes and maximum capacity
  nu = 0; mnu = nbox;  % number of U nodes and maximum capacity
  rrem = true(M,1); crem = true(N,1);  % which row/cols remain?

  % process direct interactions -- done in two loops over all boxes
  %   1. store (diagonal) self-interactions and compress near-field (if any)
  %   2. store compressed near-field interactions
  ts = tic;
  nrrem1 = nnz(rrem); ncrem1 = nnz(crem);  % remaining row/cols at start
  for lvl = 1:t.nlvl
    l = t.l(:,lvl);
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      rslf = t.nodes(i).rxi;
      cslf = t.nodes(i).cxi;

      % move on if empty
      if isempty(rslf) || isempty(cslf), continue; end

      dir = t.nodes(i).dir;
      rdir = [t.nodes(dir).rxi];
      cdir = [t.nodes(dir).cxi];

      % store self-interactions
      nb = nb + 1;
      if mnb < nb
        e = cell(mnb,1);
        s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
        F.B = [F.B; s];
        mnb = 2*mnb;
      end
      F.B(nb).is = rslf;
      if opts.symm == 'n', F.B(nb).js = cslf; end
      if opts.store ~= 'n', F.B(nb).D = A(rslf,cslf); end

      % move on if no (a priori) near-field compression
      if ~opts.near, continue; end
      if (isempty(rslf) || isempty(cdir)) && ...  % nothing to do for rows
         (isempty(rdir) || isempty(cslf))         % nothing to do for cols
         continue
       end

      % compress row space
      Kpxy = zeros(length(rslf),0);
      if isempty(pxyfun), cnbr = setdiff(find(crem),cslf);
      else, [Kpxy,cnbr] = pxyfun('r',rx,cx,rslf,cdir,l,t.nodes(i).ctr);
      end
      K = [full(A(rslf,cnbr)) Kpxy]';
      [rsk,rrd,rT] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);

      % compress column space
      if opts.symm == 'n'
        Kpxy = zeros(0,length(cslf));
        if isempty(pxyfun), rnbr = setdiff(find(rrem),rslf);
        else, [Kpxy,rnbr] = pxyfun('c',rx,cx,cslf,rdir,l,t.nodes(i).ctr);
        end
        K = [full(A(rnbr,cslf)); Kpxy];
        [csk,crd,cT] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);
      else
        csk = []; crd = []; cT = [];
      end

      % move on if no (a posteriori) compression
      if isempty(rrd) && isempty(crd), continue; end

      % store matrix factors
      nu = nu + 1;
      if mnu < nu
        e = cell(mnu,1);
        s = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
        F.U = [F.U; s];
        mnu = 2*mnu;
      end
      F.U(nu).rsk = rslf(rsk);
      F.U(nu).rrd = rslf(rrd);
      F.U(nu).csk = cslf(csk);
      F.U(nu).crd = cslf(crd);
      F.U(nu).rT = rT;
      F.U(nu).cT = cT;

      % restrict to skeletons
      t.nodes(i).rxi = rslf(rsk);
      rrem(rslf(rrd)) = false;
      if opts.symm == 'n'
        t.nodes(i).cxi = cslf(csk);
        crem(cslf(crd)) = false;
      else
        t.nodes(i).cxi = t.nodes(i).rxi;
        crem(cslf(rrd)) = false;
      end
    end
  end
  F.lvpb(2) = nb;
  F.lvpu(2) = nu;

  % store near-field interactions
  for lvl = 1:t.nlvl
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      rslf = t.nodes(i).rxi;
      cslf = t.nodes(i).cxi;
      dir = t.nodes(i).dir;
      dir = dir(dir > i);
      rdir = [t.nodes(dir).rxi];
      cdir = [t.nodes(dir).cxi];

      % move on if empty
      if (isempty(rslf) || isempty(cdir)) && (isempty(rdir) || isempty(cslf))
        continue
      end

      % store matrix factors
      nb = nb + 1;
      if mnb < nb
        e = cell(mnb,1);
        s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
        F.B = [F.B; s];
        mnb = 2*mnb;
      end
      F.B(nb).is = rslf;
      F.B(nb).ie = rdir;
      if opts.symm == 'n'
        F.B(nb).js = cslf;
        F.B(nb).je = cdir;
      end
      if opts.store == 'r' || opts.store == 'a'
        F.B(nb).Bo = A(rdir,cslf);
        if opts.symm == 'n', F.B(nb).Bi = A(rslf,cdir); end
      end
    end
  end
  F.lvpb(3) = nb;

  % print summary
  if opts.verb
    nrrem2 = nnz(rrem); ncrem2 = nnz(crem);  % remaining row/cols at end
    nblk = pblk(t.nlvl+1);  % total number of nonempty boxes
    fprintf('%3s | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
            'n',nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,toc(ts))
    fprintf('%3s | %6s | %8d | %8d | %8.2f | %8.2f | %10s\n', ...
            ' ',' ',ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
  end

  % loop over tree levels
  nlvl = 1;  % offset for extra near-field level
  for lvl = t.nlvl:-1:1
    ts = tic;
    nlvl = nlvl + 1;
    nrrem1 = nnz(rrem); ncrem1 = nnz(crem);  % remaining row/cols at start
    l = t.l(:,lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).rxi = [t.nodes(i).rxi [t.nodes(t.nodes(i).chld).rxi]];
      t.nodes(i).cxi = [t.nodes(i).cxi [t.nodes(t.nodes(i).chld).cxi]];
    end

    % loop over nodes
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      rslf = t.nodes(i).rxi;
      cslf = t.nodes(i).cxi;
      rnbr = [t.nodes(t.nodes(i).nbor).rxi];
      cnbr = [t.nodes(t.nodes(i).nbor).cxi];

      % compress row space
      if lvl > 2
        if isempty(pxyfun)
          cfar = setdiff(find(crem),[cslf cnbr]);
          K = A(rslf,cfar);
        else
          K = pxyfun('r',rx,cx,rslf,cnbr,l,t.nodes(i).ctr);
        end
      else
        K = zeros(length(rslf),0);
      end
      [rsk,rrd,rT] = id(K',rank_or_tol,opts.Tmax,opts.rrqr_iter);

      % compress column space
      if opts.symm == 'n'
        if lvl > 2
          if isempty(pxyfun)
            rfar = setdiff(find(rrem),[rslf rnbr]);
            K = A(rfar,cslf);
          else
            K = pxyfun('c',rx,cx,cslf,rnbr,l,t.nodes(i).ctr);
          end
        else
          K = zeros(0,length(cslf));
        end
        [csk,crd,cT] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);
      else
        csk = []; crd = []; cT = [];
      end

      % move on if no compression
      if isempty(rrd) && isempty(crd), continue; end

      % store matrix factors
      nu = nu + 1;
      if mnu < nu
        e = cell(mnu,1);
        s = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
        F.U = [F.U; s];
        mnu = 2*mnu;
      end
      F.U(nu).rsk = rslf(rsk);
      F.U(nu).rrd = rslf(rrd);
      F.U(nu).csk = cslf(csk);
      F.U(nu).crd = cslf(crd);
      F.U(nu).rT = rT;
      F.U(nu).cT = cT;

      % restrict to skeletons for next level
      t.nodes(i).rxi = rslf(rsk);
      rrem(rslf(rrd)) = false;
      if opts.symm == 'n'
        t.nodes(i).cxi = cslf(csk);
        crem(cslf(crd)) = false;
      else
        t.nodes(i).cxi = t.nodes(i).rxi;
        crem(cslf(rrd)) = false;
      end
    end
    F.lvpu(nlvl+1) = nu;

    % store far-field interactions
    if lvl > 2
      for i = t.lvp(lvl)+1:t.lvp(lvl+1)
        rslf = t.nodes(i).rxi;
        cslf = t.nodes(i).cxi;

        % generate interaction list
        ilst = [];
        pnbor = t.nodes(t.nodes(i).prnt).nbor;
        for j = pnbor
          % include nonempty parent-neighbors
          if ~isempty([t.nodes(j).rxi t.nodes(j).cxi]), ilst = [ilst j]; end
          % include their children if at same level as parent
          if j > t.lvp(lvl-1), ilst = [ilst t.nodes(j).chld]; end
        end
        % remove neighbors
        ilst_sort = sort(ilst);
        ilst = ilst_sort(~ismemb(ilst_sort,sort(t.nodes(i).nbor)));
        % keep if at higher level; avoid double-counting at same level
        ilst = ilst(ilst <= t.lvp(lvl) | (ilst > t.lvp(lvl) & ilst > i));
        rint = [t.nodes(ilst).rxi];
        cint = [t.nodes(ilst).cxi];

        % move on if empty
        if (isempty(rslf) || isempty(cint)) && (isempty(cslf) || isempty(rint))
          continue
        end

        % store matrix factors
        nb = nb + 1;
        if mnb < nb
          e = cell(mnb,1);
          s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
          F.B = [F.B; s];
          mnb = 2*mnb;
        end
        F.B(nb).is = rslf;
        F.B(nb).ie = rint;
        if opts.symm == 'n'
          F.B(nb).js = cslf;
          F.B(nb).je = cint;
        end
        if opts.store == 'a'
          F.B(nb).Bo = A(rint,cslf);
          if opts.symm == 'n', F.B(nb).Bi = A(rslf,cint); end
        end
      end
    end
    F.lvpb(nlvl+2) = nb;

    % print summary
    if opts.verb
      nrrem2 = nnz(rrem); ncrem2 = nnz(crem);        % remaining row/cols at end
      nblk = pblk(lvl) + t.lvp(lvl+1) - t.lvp(lvl);  % nonempty up to this level
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
              lvl,nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,toc(ts))
      fprintf('%3s | %6s | %8d | %8d | %8.2f | %8.2f | %10s\n', ...
              ' ',' ',ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
    end
  end

  % finish
  F.B = F.B(1:nb);
  F.U = F.U(1:nu);
  if opts.verb, fprintf([repmat('-',1,69) '\n']), end
end