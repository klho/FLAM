% RSKEL  Recursive skeletonization.
%
%    The recursive skeletonization algorithm approximates a hierarchically rank-
%    structured matrix A as a multilevel low-rank perturbation A = D + U*S*V',
%    where D is a block diagonal submatrix of A, U and V are block diagonal and
%    rectangular, and S is a submatrix of A - D and is itself approximated in
%    the same form, e.g., A = D1 + U1*(D2 + U2*S*V2')*V1'.
%
%    The matrix A can be rectangular. This representation facilitates fast
%    multiplication, inversion, and least squares, among other operations.
%
%    Each row/column of A is associated with a point, with the induced point
%    geometry exposing the required rank structure. In typical operation, only
%    the near-field (near-diagonal blocks) is explicitly evaluated; the far-
%    field is captured by a user-supplied "proxy" function. To avoid excessive
%    storage, the matrix should be given as a function handle implementing the
%    usual submatrix access interface.
%
%    Typical complexity for [M,N] = SIZE(A) with M >= N without loss of
%    generality: O(M + N) in 1D and O(M + N^(3*(1 - 1/D))) in D dimensions.
%
%    F = RSKEL(A,RX,CX,OCC,RANK_OR_TOL) produces a compressed representation F
%    of the matrix A acting on the row and column points RX and CX,
%    respectively, using tree occupancy parameter OCC and local precision
%    parameter RANK_OR_TOL. See HYPOCT and ID for details. Note that both row/
%    column points are sorted together in the same tree, so OCC should be set
%    roughly twice the desired leaf size. Since no proxy function is supplied,
%    this simply performs a naive compression of all off-diagonal blocks.
%
%    F = RSKEL(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) accelerates the compression using
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
%    responsible for handling them. See the examples for further details. If
%    PXYFUN is not provided or empty (default), then the code uses the naive
%    global compression scheme.
%
%    F = RSKEL(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options
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
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N'). Symmetry
%              can reduce the computation time by about a factor of two.
%
%      - VERB: display status info if VERB = 1 (default: VERB = 0). This prints
%              to screen a table tracking row/column compression statistics
%              through level. Special levels: 'T', tree sorting.
%
%    Primary references:
%
%      K.L. Ho, L. Greengard. A fast direct solver for structured linear systems
%        by recursive skeletonization. SIAM J. Sci. Comput. 34 (5): A2507-A2532,
%        2012.
%
%      K.L. Ho, L. Greengard. A fast semidirect least squares algorithm for
%        hierarchically block separable matrices. SIAM J. Matrix Anal. Appl. 35
%        (2): 725-748, 2014.
%
%    Other references:
%
%      S. Chandrasekaran, P. Dewilde, M. Gu, W. Lyons, T. Pals. A fast solver
%        for HSS representations via sparse matrices. SIAM J. Matrix Anal. Appl.
%        29 (1): 67-81, 2006.
%
%      A. Gillman, P.M. Young, P.-G. Martinsson. A direct solver with O(N)
%        complexity for integral equations on one-dimensional domains. Front.
%        Math. China 7 (2): 217-247, 2012.
%
%      P.G. Martinsson, V. Rokhlin. A fast direct solver for boundary integral
%        equations in two dimensions. J. Comput. Phys. 205 (1): 1-23, 2005.
%
%    See also HYPOCT, ID, RSKEL_MV, RSKEL_XSP.

function F = rskel(A,rx,cx,occ,rank_or_tol,pxyfun,opts)

  % set default parameters
  if nargin < 6, pxyfun = []; end
  if nargin < 7, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'ext'), opts.ext = []; end
  if ~isfield(opts,'Tmax'), opts.Tmax = 2; end
  if ~isfield(opts,'rrqr_iter'), opts.rrqr_iter = Inf; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
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

  % initialize
  nbox = t.lvp(end);
  e = cell(nbox,1);
  D = struct('i',e,'j',e,'D',e);  % diagonal submatrix entries
  U = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);  % ID matrices
  F = struct('M',M,'N',N,'P',P,'Q',Q,'nlvl',t.nlvl,'lvpd',zeros(1,t.nlvl+1), ...
             'lvpu',zeros(1,t.nlvl+1),'D',D,'U',U,'symm',opts.symm);
  nlvl = 0;
  nd = 0;
  nu = 0;
  rrem = true(M,1); crem = true(N,1);  % which row/cols remain?

  % loop over tree levels
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

      % generate diagonal block
      if isempty(t.nodes(i).chld)
        rxi = rslf;
        cxi = cslf;
        if isempty(rxi) || isempty(cxi), continue; end
        nd = nd + 1;
        F.D(nd).i = rxi;
        F.D(nd).j = cxi;
        F.D(nd).D = A(rxi,cxi);
      else
        % for non-leaf, avoid storing zero diagonal blocks from children
        chld = t.nodes(i).chld;
        for k = chld
          j = chld(chld ~= k);
          rxi = [t.nodes(j).rxi];
          cxi = t.nodes(k).cxi;
          if isempty(rxi) || isempty(cxi), continue; end
          nd = nd + 1;
          F.D(nd).i = rxi;
          F.D(nd).j = cxi;
          F.D(nd).D = A(rxi,cxi);
        end
      end

      % compress row space
      Kpxy = zeros(length(rslf),0);
      if lvl > 2
        if isempty(pxyfun), cnbr = setdiff(find(crem),cslf);
        else, [Kpxy,cnbr] = pxyfun('r',rx,cx,rslf,cnbr,l,t.nodes(i).ctr);
        end
      end
      K = [full(A(rslf,cnbr)) Kpxy]';
      [rsk,rrd,rT] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);

      % compress column space
      if opts.symm == 'n'
        Kpxy = zeros(0,length(cslf));
        if lvl > 2
          if isempty(pxyfun), rnbr = setdiff(find(rrem),rslf);
          else, [Kpxy,rnbr] = pxyfun('c',rx,cx,cslf,rnbr,l,t.nodes(i).ctr);
          end
        end
        K = [full(A(rnbr,cslf)); Kpxy];
        [csk,crd,cT] = id(K,rank_or_tol,opts.Tmax,opts.rrqr_iter);
      else
        csk = []; crd = []; cT = [];
      end

      % move on if no compression
      if isempty(rrd) && isempty(crd), continue; end

      % store matrix factors
      nu = nu + 1;
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
    F.lvpd(nlvl+1) = nd;
    F.lvpu(nlvl+1) = nu;
    te = toc(ts);

    % print summary
    if opts.verb
      nrrem2 = nnz(rrem); ncrem2 = nnz(crem);        % remaining row/cols at end
      nblk = pblk(lvl) + t.lvp(lvl+1) - t.lvp(lvl);  % nonempty up to this level
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
              lvl,nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,te)
      fprintf('%3s | %6s | %8d | %8d | %8.2f | %8.2f | %10s\n', ...
              ' ',' ',ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
    end
  end

  % finish
  F.D = F.D(1:nd);
  F.U = F.U(1:nu);
  if opts.verb, fprintf([repmat('-',1,69) '\n']), end
end