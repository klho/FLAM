% RSKEL  Recursive skeletonization.
%
%    F = RSKEL(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) produces a compressed
%    representation F of the interaction matrix A on the row and column points
%    RX and CX, respectively, using tree occupancy parameter OCC, local
%    precision parameter RANK_OR_TOL, and proxy function PXYFUN to capture the
%    far field. This is a function of the form
%
%      [KPXY,NBR] = PXYFUN(RC,RX,CX,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor indices (can be modified)
%      - RC:   flag to specify row or column compression ('R' or 'C')
%      - RX:   input row points
%      - CX:   input column points
%      - SLF:  block indices
%      - L:    block size
%      - CTR:  block center
%
%    See the examples for further details. If PXYFUN is not provided or empty
%    (default), then the code uses the naive global compression scheme.
%
%    F = RSKEL(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options
%    to the algorithm. Valid options include:
%
%      - EXT: set the root node extent to [EXT(I,1) EXT(I,2)] along dimension I.
%             If EXT is empty (default), then the root extent is calculated from
%             the data.
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N').
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      S. Chandrasekaran, P. Dewilde, M. Gu, W. Lyons, T. Pals. A fast solver
%        HSS representations via sparse matrices. SIAM J. Matrix Anal. Appl.
%        29 (1): 67-81, 2006.
%
%      A. Gillman, P.M. Young, P.-G. Martinsson. A direct solver with O(N)
%        complexity for integral equations on one-dimensional domains. Front.
%        Math. China 7 (2): 217-247, 2012.
%
%      K.L. Ho, L. Greengard. A fast direct solver for structured linear systems
%        by recursive skeletonization. SIAM J. Sci. Comput. 34 (5): A2507-A2532,
%        2012.
%
%      K.L. Ho, L. Greengard. A fast semidirect least squares algorithm for
%        hierarchically block separable matrices. SIAM J. Matrix Anal. Appl. 35
%        (2): 725--748, 2014.
%
%      P.G. Martinsson, V. Rokhlin. A fast direct solver for boundary integral
%        equations in two dimensions. J. Comput. Phys. 205 (1): 1-23, 2005.
%
%    See also HYPOCT, ID, RSKEL_DIAGS, RSKEL_MV, RSKEL_XSP.

function F = rskel(A,rx,cx,occ,rank_or_tol,pxyfun,opts)
  start = tic;

  % set default parameters
  if nargin < 6
    pxyfun = [];
  end
  if nargin < 7
    opts = [];
  end
  if ~isfield(opts,'ext')
    opts.ext = [];
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
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:rskel:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')

  % build tree
  M = size(rx,2);
  N = size(cx,2);
  tic
  t = hypoct([rx cx],occ,opts.lvlmax,opts.ext);
  for i = 1:t.lvp(t.nlvl+1)
    xi = t.nodes(i).xi;
    idx = xi <= M;
    t.nodes(i).rxi = xi( idx);
    t.nodes(i).cxi = xi(~idx) - M;
    t.nodes(i).xi = [];
  end

  % print summary
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    fprintf('%3s | %63.2e (s)\n','-',toc)

    % count nonempty boxes at each level
    pblk = zeros(t.nlvl+1,1);
    for lvl = 1:t.nlvl
      pblk(lvl+1) = pblk(lvl);
      for i = t.lvp(lvl)+1:t.lvp(lvl+1)
        if ~isempty([t.nodes(i).rxi t.nodes(i).cxi])
          pblk(lvl+1) = pblk(lvl+1) + 1;
        end
      end
    end
  end

  % initialize
  nbox = t.lvp(end);
  e = cell(nbox,1);
  D = struct('i',e,'j',e,'D',e);
  U = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
  F = struct('M',M,'N',N,'nlvl',t.nlvl,'lvpd',zeros(1,t.nlvl+1),'lvpu', ...
             zeros(1,t.nlvl+1),'D',D,'U',U,'symm',opts.symm);
  nlvl = 0;
  nd = 0;
  nu = 0;
  rrem = true(M,1);
  crem = true(N,1);

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    tic
    nlvl = nlvl + 1;
    nrrem1 = sum(rrem);
    ncrem1 = sum(crem);
    l = t.lrt/2^(lvl - 1);

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
        if ~isempty(rxi) && ~isempty(cxi)
          nd = nd + 1;
          F.D(nd).i = rxi;
          F.D(nd).j = cxi;
          F.D(nd).D = A(rxi,cxi);
        end
      else
        chld = t.nodes(i).chld;
        for k = chld
          j = chld(chld ~= k);
          rxi = [t.nodes(j).rxi];
          cxi = t.nodes(k).cxi;
          if ~isempty(rxi) && ~isempty(cxi)
            nd = nd + 1;
            F.D(nd).i = rxi;
            F.D(nd).j = cxi;
            F.D(nd).D = A(rxi,cxi);
          end
        end
      end

      % compress row space
      Kpxy = zeros(length(rslf),0);
      if lvl > 2
        if isempty(pxyfun)
          cnbr = setdiff(find(crem),cslf);
        else
          [Kpxy,cnbr] = pxyfun('r',rx,cx,rslf,cnbr,l,t.nodes(i).ctr);
        end
      end
      K = full(A(rslf,cnbr));
      K = [K Kpxy]';
      [rsk,rrd,rT] = id(K,rank_or_tol);

      % compress column space
      if strcmpi(opts.symm,'n')
        Kpxy = zeros(0,length(cslf));
        if lvl > 2
          if isempty(pxyfun)
            rnbr = setdiff(find(rrem),rslf);
          else
            [Kpxy,rnbr] = pxyfun('c',rx,cx,cslf,rnbr,l,t.nodes(i).ctr);
          end
        end
        K = full(A(rnbr,cslf));
        K = [K; Kpxy];
        [csk,crd,cT] = id(K,rank_or_tol);
      else
        csk = [];
        crd = [];
        cT  = [];
      end

      % move on if no compression
      if isempty(rrd) && isempty(crd)
        continue
      end

      % store matrix factors
      nu = nu + 1;
      F.U(nu).rsk = rslf(rsk);
      F.U(nu).rrd = rslf(rrd);
      F.U(nu).csk = cslf(csk);
      F.U(nu).crd = cslf(crd);
      F.U(nu).rT = rT';
      F.U(nu).cT = cT;

      % restrict to skeletons
      t.nodes(i).rxi = rslf(rsk);
      rrem(rslf(rrd)) = 0;
      if strcmpi(opts.symm,'n')
        t.nodes(i).cxi = cslf(csk);
        crem(cslf(crd)) = 0;
      else
        t.nodes(i).cxi = t.nodes(i).rxi;
        crem(cslf(rrd)) = 0;
      end
    end
    F.lvpd(nlvl+1) = nd;
    F.lvpu(nlvl+1) = nu;

    % print summary
    if opts.verb
      nrrem2 = sum(rrem);
      ncrem2 = sum(crem);
      nblk = pblk(lvl) + t.lvp(lvl+1) - t.lvp(lvl);
      fprintf('%3d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
              lvl,nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,toc)
      fprintf('%3s | %6s | %8d | %8d | %8.2f | %8.2f | %10s (s)\n', ...
              ' ',' ',ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
    end
  end

  % finish
  F.D = F.D(1:nd);
  F.U = F.U(1:nu);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end