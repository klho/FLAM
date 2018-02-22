% RSKELFR  Recursive skeletonization factorization for rectangular matrices.
%
%    F = RSKELFR(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) produces a factorization F of
%    the interaction matrix A on the row and column points RX and CX,
%    respectively, using tree occupancy parameter OCC, local precision parameter
%    RANK_OR_TOL, and proxy function PXYFUN to capture the far field. This is a
%    function of the form
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
%    F = RSKELFR(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various
%    options to the algorithm. Valid options include:
%
%      - EXT: set the root node extent to [EXT(I,1) EXT(I,2)] along dimension I.
%             If EXT is empty (default), then the root extent is calculated from
%             the data.
%
%      - LVLMAX: maximum tree depth (default: LVLMAX = Inf).
%
%      - RDPIV: pivoting strategy for redundant point subselection. No pivoting
%               is used if RDPIV = 'N', LU pivoting if RDPIV = 'L', and QR
%               pivoting if RDPIV = 'Q' (default: RDPIV = 'L').
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. Comm. Pure Appl. Math. 69 (7):
%        1314-1353, 2016.
%
%    See also HYPOCT, ID, RSKELFR_MV, RSKELFR_SV.

function F = rskelfr(A,rx,cx,occ,rank_or_tol,pxyfun,opts)
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
  if ~isfield(opts,'rdpiv')
    opts.rdpiv = 'l';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(strcmpi(opts.rdpiv,'n') || strcmpi(opts.rdpiv,'l') || ...
         strcmpi(opts.rdpiv,'q'), ...
         'FLAM:rskelfr:invalidRdpiv', ...
         'Redundant pivoting parameter must be one of ''N'', ''L'', or ''Q''.')

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
  F = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e,'E',e,'F',e,'L', ...
             e,'U',e);
  F = struct('M',M,'N',N,'nlvl',t.nlvl,'lvp',zeros(1,t.nlvl+1),'factors',F);
  nlvl = 0;
  n = 0;
  rrem = true(M,1);
  crem = true(N,1);
  S = cell(nbox,1);
  rI = zeros(M,1);
  cI = zeros(N,1);

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
      nrslf = length(rslf);
      ncslf = length(cslf);

      % generate modified diagonal block
      S{i} = full(A(rslf,cslf));
      if lvl < t.nlvl
        rI(rslf) = 1:nrslf;
        cI(cslf) = 1:ncslf;
        for j = t.nodes(i).chld
          rxi = t.nodes(j).rxi;
          cxi = t.nodes(j).cxi;
          S{i}(rI(rxi),cI(cxi)) = S{j};
          S{j} = [];
        end
      end

      % skeletonize rows
      Kpxy = zeros(nrslf,0);
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

      % skeletonize columns
      Kpxy = zeros(0,ncslf);
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

      % move on if no compression
      if isempty(rrd) && isempty(crd)
        continue
      end

      % find good redundant pivots
      K = S{i};
      if lvl > 1
        nrrd = length(rrd);
        ncrd = length(crd);
        if nrrd > ncrd
          [rsk,rrd,rT] = rdpivot('r',K(rrd,crd),rsk,rrd,rT);
        elseif nrrd < ncrd
          [csk,crd,cT] = rdpivot('c',K(rrd,crd),csk,crd,cT);
        end
      end

      % compute factors
      K(rrd,:) = K(rrd,:) - rT'*K(rsk,:);
      K(:,crd) = K(:,crd) - K(:,csk)*cT;
      [L,U] = lu(K(rrd,crd));
      E = (K(rsk,crd)/U)/L;
      G = U\(L\K(rrd,csk));

      % update self-interaction
      S{i} = S{i}(rsk,csk) - E*(L*(U*G));

      % store matrix factors
      n = n + 1;
      F.factors(n).rsk = rslf(rsk);
      F.factors(n).rrd = rslf(rrd);
      F.factors(n).csk = cslf(csk);
      F.factors(n).crd = cslf(crd);
      F.factors(n).rT = rT';
      F.factors(n).cT = cT;
      F.factors(n).E = E;
      F.factors(n).F = G;
      F.factors(n).L = L;
      F.factors(n).U = U;

      % restrict to skeletons
      t.nodes(i).rxi = rslf(rsk);
      t.nodes(i).cxi = cslf(csk);
      rrem(rslf(rrd)) = 0;
      crem(cslf(crd)) = 0;
    end
    F.lvp(nlvl+1) = n;

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
  F.factors = F.factors(1:n);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end

  % pivoting for redundant subselection
  function [sk,rd,T] = rdpivot(rc,X,sk,rd,T)
    [m_,n_] = size(X);
    k = min(m_,n_);
    if k > 0
      if strcmpi(rc,'r')
        if strcmpi(opts.rdpiv,'n')
          p = 1:m_;
        elseif strcmpi(opts.rdpiv,'l')
          [~,~,p] = lu(X,'vector');
        elseif strcmpi(opts.rdpiv,'q')
          [~,~,p] = qr(X','vector');
        end
      elseif strcmpi(rc,'c')
        if strcmpi(opts.rdpiv,'n')
          p = 1:n_;
        elseif strcmpi(opts.rdpiv,'l')
          [~,~,p] = lu(X','vector');
        elseif strcmpi(opts.rdpiv,'q')
          [~,~,p] = qr(X,'vector');
        end
      end
      sk = [sk rd(p(k+1:end))];
      idx = p(1:k);
      rd = rd(idx);
      T = [T(:,idx); zeros(length(sk)-size(T,1),length(rd))];
    else
      sk = [sk rd];
      rd = [];
      T = zeros(length(sk),length(rd));
    end
  end
end