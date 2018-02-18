% IFMM   Interpolative fast multipole method.
%
%    F = IFMM(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) produces a compressed
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
%    F = IFMM(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options
%    to the algorithm. Valid options include:
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - NEAR: compress near field if NEAR = 1 (default: NEAR = 0).
%
%      - STORE: store no interactions if STORE = 'N', only self-interactions if
%               if STORE = 'S', only near-field (including self-) interactions
%               if STORE = 'R', and all interactions if STORE = 'A' (default:
%               STORE = 'N').
%
%      - SYMM: assume that the matrix is unsymmetric if SYMM = 'N', (complex-)
%              symmetric if SYMM = 'S', Hermitian if SYMM = 'H', and Hermitian
%              positive definite if SYMM = 'P' (default: SYMM = 'N').
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      J. Carrier, L. Greengard, V. Rokhlin. A fast adaptive multipole algorithm
%        for particle simulations. SIAM J. Sci. Stat. Comput. 9 (4): 669-686,
%        1998.
%
%      P.G. Martinsson, V. Rokhlin. An accelerated kernel-independent fast
%        multipole method in one dimension. SIAM J. Sci. Comput. 29 (3):
%        1160-1178, 2007.
%
%      X. Pan, X. Sheng. Hierarchical interpolative decomposition multilevel
%        fast multipole algorithm for dynamic electromagnetic simulations. Prog.
%        Electromag. Res. 134: 79-94, 2013.
%
%    See also HYPOCT, ID, IFMM_MV.

function F = ifmm(A,rx,cx,occ,rank_or_tol,pxyfun,opts)
  start = tic;

  % set default parameters
  if nargin < 6
    opts = [];
  end
  if ~isfield(opts,'lvlmax')
    opts.lvlmax = Inf;
  end
  if ~isfield(opts,'near')
    opts.near = 0;
  end
  if ~isfield(opts,'store')
    opts.store = 'n';
  end
  if ~isfield(opts,'symm')
    opts.symm = 'n';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(strcmpi(opts.store,'n') || strcmpi(opts.store,'s') || ...
         strcmpi(opts.store,'r') || strcmpi(opts.store,'a'), ...
         'FLAM:ifmm:invalidStore', ...
         'Storage parameter must be one of ''N'', ''S'', ''R'', or ''A''.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:ifmm:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')

  % build tree
  M = size(rx,2);
  N = size(cx,2);
  tic
  t = hypoct([rx cx],occ,opts.lvlmax);
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

  % find direct interactions
  for i = 1:t.lvp(end)
    t.nodes(i).dir = [];
  end
  for lvl = 1:t.nlvl
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      if isempty([t.nodes(i).rxi t.nodes(i).cxi])
        continue
      end
      for j = t.nodes(i).nbor
        if isempty([t.nodes(j).rxi t.nodes(j).cxi])
          continue
        end
          t.nodes(i).dir = [t.nodes(i).dir j];
        if j <= t.lvp(lvl)
          t.nodes(j).dir = [t.nodes(j).dir i];
        end
      end
    end
  end

  % initialize
  nlvl = t.nlvl + 1;
  nbox = t.lvp(end);
  mnb = nbox;
  mnu = nbox;
  e = cell(nbox,1);
  B = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
  U = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
  F = struct('M',M,'N',N,'nlvl',nlvl,'lvpb',zeros(1,nlvl+1),'lvpu', ...
             zeros(1,nlvl+1),'B',B,'U',U,'store',opts.store,'symm',opts.symm);
  nb = 0;
  nu = 0;
  rrem = true(M,1);
  crem = true(N,1);

  % process direct interactions
  tic
  nrrem1 = sum(rrem);
  ncrem1 = sum(crem);
  for lvl = 1:t.nlvl
    l = t.lrt/2^(lvl - 1);
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      rslf = t.nodes(i).rxi;
      cslf = t.nodes(i).cxi;
      dir = t.nodes(i).dir;
      rdir = [t.nodes(dir).rxi];
      cdir = [t.nodes(dir).cxi];

      % move on if empty
      if isempty(rslf) || isempty(cslf)
        continue
      end

      % store self-interactions
      nb = nb + 1;
      while mnb < nb
        e = cell(mnb,1);
        s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
        F.B = [F.B; s];
        mnb = 2*mnb;
      end
      F.B(nb).is = rslf;
      if strcmpi(opts.symm,'n')
        F.B(nb).js = cslf;
      end
      if strcmpi(opts.store,'s') || strcmpi(opts.store,'r') || ...
         strcmpi(opts.store,'a')
        F.B(nb).D = A(rslf,cslf);
      end

      % move on if no (a priori) near-field compression
      if ~opts.near || (isempty(rdir) || isempty(cslf)) && ...
                       (isempty(rslf) || isempty(cdir))
        continue
      end

      % compress row space
      Kpxy = zeros(length(rslf),0);
      if isempty(pxyfun)
        cnbr = setdiff(find(crem),cslf);
      else
        [Kpxy,cnbr] = pxyfun('r',rx,cx,rslf,cdir,l,t.nodes(i).ctr);
      end
      K = full(A(rslf,cnbr));
      K = [K Kpxy]';
      [rsk,rrd,rT] = id(K,rank_or_tol);

      % compress column space
      if strcmpi(opts.symm,'n')
        Kpxy = zeros(0,length(cslf));
        if isempty(pxyfun)
          rnbr = setdiff(find(rrem),rslf);
        else
          [Kpxy,rnbr] = pxyfun('c',rx,cx,cslf,rdir,l,t.nodes(i).ctr);
        end
        K = full(A(rnbr,cslf));
        K = [K; Kpxy];
        [csk,crd,cT] = id(K,rank_or_tol);
      else
        csk = [];
        crd = [];
        cT  = [];
      end

      % move on if no (a posteriori) compression
      if isempty(rrd) && isempty(crd)
        continue
      end

      % store matrix factors
      nu = nu + 1;
      while mnu < nu
        e = cell(mnu,1);
        s = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
        F.U = [F.U; s];
        mnu = 2*mnu;
      end
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
      if (isempty(rdir) || isempty(cslf)) && (isempty(rslf) || isempty(cdir))
        continue
      end

      % store matrix factors
      nb = nb + 1;
      while mnb < nb
        e = cell(mnb,1);
        s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
        F.B = [F.B; s];
        mnb = 2*mnb;
      end
      F.B(nb).is = rslf;
      F.B(nb).ie = rdir;
      if strcmpi(opts.symm,'n')
        F.B(nb).js = cslf;
        F.B(nb).je = cdir;
      end
      if strcmpi(opts.store,'r') || strcmpi(opts.store,'a')
        F.B(nb).Bo = A(rdir,cslf);
        if strcmpi(opts.symm,'n')
          F.B(nb).Bi = A(rslf,cdir);
        end
      end
    end
  end
  F.lvpb(3) = nb;

  % print summary
  if opts.verb
    nrrem2 = sum(rrem);
    ncrem2 = sum(crem);
    nblk = pblk(t.nlvl+1);
    fprintf('%3s | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
            '*',nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,toc)
    fprintf('%3s | %6s | %8d | %8d | %8.2f | %8.2f | %10s (s)\n', ...
            ' ',' ',ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
  end

  % loop over tree levels
  nlvl = 1;
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

      % compress row space
      if lvl > 2 && ~isempty(pxyfun)
        K = pxyfun('r',rx,cx,rslf,cnbr,l,t.nodes(i).ctr);
      else
        cfar = setdiff(find(crem),[cslf cnbr]);
        K = A(rslf,cfar);
      end
      [rsk,rrd,rT] = id(K',rank_or_tol);

      % compress column space
      if strcmpi(opts.symm,'n')
        if lvl > 2 && ~isempty(pxyfun)
          K = pxyfun('c',rx,cx,cslf,rnbr,l,t.nodes(i).ctr);
        else
          rfar = setdiff(find(rrem),[rslf rnbr]);
          K = A(rfar,cslf);
        end
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
      while mnu < nu
        e = cell(mnu,1);
        s = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e);
        F.U = [F.U; s];
        mnu = 2*mnu;
      end
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
    F.lvpu(nlvl+1) = nu;

    % process far-field interactions
    if lvl > 2
      for i = t.lvp(lvl)+1:t.lvp(lvl+1)
        rslf = t.nodes(i).rxi;
        cslf = t.nodes(i).cxi;

        % generate interaction list
        ilst = [];
        pnbor = t.nodes(t.nodes(i).prnt).nbor;
        for j = pnbor
          if ~isempty([t.nodes(j).rxi t.nodes(j).cxi])
            ilst = [ilst j];
          end
          if j > t.lvp(lvl-1)
            ilst = [ilst t.nodes(j).chld];
          end
        end
        ilst_sort = sort(ilst);
        ilst = ilst_sort(~ismemb(ilst_sort,sort(t.nodes(i).nbor)));
        ilst = ilst(ilst <= t.lvp(lvl) | (ilst > t.lvp(lvl) & ilst > i));
        rint = [t.nodes(ilst).rxi];
        cint = [t.nodes(ilst).cxi];

        % move on if empty
        if (isempty(rint) || isempty(cslf)) && (isempty(cint) || isempty(rslf))
          continue
        end

        % store matrix factors
        nb = nb + 1;
        while mnb < nb
          e = cell(mnb,1);
          s = struct('is',e,'js',e,'ie',e,'je',e,'D',e,'Bo',e,'Bi',e);
          F.B = [F.B; s];
          mnb = 2*mnb;
        end
        F.B(nb).is = rslf;
        F.B(nb).ie = rint;
        if strcmpi(opts.symm,'n')
          F.B(nb).js = cslf;
          F.B(nb).je = cint;
        end
        if strcmpi(opts.store,'a')
          F.B(nb).Bo = A(rint,cslf);
          if strcmpi(opts.symm,'n')
            F.B(nb).Bi = A(rslf,cint);
          end
        end
      end
      F.lvpb(nlvl+2) = nb;
    else
      F.lvpb(nlvl+1) = nb;
    end

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
  F.B = F.B(1:nb);
  F.U = F.U(1:nu);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end