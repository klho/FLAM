% HIFIE2R  Hierarchical interpolative factorization for rectangular integral
%          operators in 2D.
%
%    F = HIFIE2R(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN) produces a factorization F of
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
%    F = HIFIE2R(A,RX,CX,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various
%    options to the algorithm. Valid options include:
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
%    See also HIFIER_MV, HIFIER_SV, HYPOCT, ID.

function F = hifie2r(A,rx,cx,occ,rank_or_tol,pxyfun,opts)
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
  if ~isfield(opts,'skip')
    opts.skip = 0;
  end
  if ~isfield(opts,'rdpiv')
    opts.rdpiv = 'l';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(opts.skip >= 0,'FLAM:hifie2r:negativeSkip', ...
         'Skip parameter must be nonnegative.')
  assert(strcmpi(opts.rdpiv,'n') || strcmpi(opts.rdpiv,'l') || ...
         strcmpi(opts.rdpiv,'q'), ...
         'FLAM:hifie2r:invalidRdpiv', ...
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
    fprintf(' %3s  | %63.2e (s)\n','-',toc)

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
  F = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e,'E',e,'F',e,'L', ...
             e,'U',e);
  F = struct('M',M,'N',N,'nlvl',t.nlvl,'lvp',zeros(1,t.nlvl+1),'factors',F);
  nlvl = 0;
  n = 0;
  rrem = true(M,1);   % which indices not yet eliminated
  crem = true(N,1);
  rcrem = true(M,1);  % which indices need to be considered for ID
  ccrem = true(N,1);
  mnz = 128;
  X = sparse(M,N);
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  S = zeros(mnz,1);
  rP = zeros(M,1);
  cP = zeros(N,1);

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    l = t.lrt/2^(lvl - 1);
    nbox = t.lvp(lvl+1) - t.lvp(lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      t.nodes(i).rxi = [t.nodes(i).rxi [t.nodes(t.nodes(i).chld).rxi]];
      t.nodes(i).cxi = [t.nodes(i).cxi [t.nodes(t.nodes(i).chld).cxi]];
    end

    % loop over dimensions
    for d = [2 1]
      tic

      % dimension reduction
      if d < 2

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
        blocks = struct('ctr',e,'rxi',e,'cxi',e,'rprnt',e,'cprnt',e, ...
                        'nbr1',e,'nbr2',e);

        % sort points by centers
        for box = 1:nbox
          % rows
          xi = [t.nodes(t.lvp(lvl)+box).rxi];
          i = box2ctr{box};
          dx = bsxfun(@minus,rx(1,xi),ctr(i,1));
          dy = bsxfun(@minus,rx(2,xi),ctr(i,2));
          dist = sqrt(dx.^2 + dy.^2);
          near = bsxfun(@eq,dist,min(dist,[],1));
          for i = 1:length(xi)
            rP(xi(i)) = box2ctr{box}(find(near(:,i),1));
          end
          % columns
          xi = [t.nodes(t.lvp(lvl)+box).cxi];
          i = box2ctr{box};
          dx = bsxfun(@minus,cx(1,xi),ctr(i,1));
          dy = bsxfun(@minus,cx(2,xi),ctr(i,2));
          dist = sqrt(dx.^2 + dy.^2);
          near = bsxfun(@eq,dist,min(dist,[],1));
          for i = 1:length(xi)
            cP(xi(i)) = box2ctr{box}(find(near(:,i),1));
          end
        end
        for box = 1:nbox
          pbox = t.lvp(lvl) + box;
          % rows
          xi = [t.nodes(pbox).rxi];
          if ~isempty(xi)
            m = histc(rP(xi),1:nb);
            p = cumsum(m);
            p = [0; p(:)];
            [~,idx] = sort(rP(xi));
            xi = xi(idx);
            for j = box2ctr{box}
              blocks(j).rxi = [blocks(j).rxi xi(p(j)+1:p(j+1))];
              blocks(j).rprnt = [blocks(j).rprnt pbox*ones(1,m(j))];
            end
          end
          % columns
          xi = [t.nodes(pbox).cxi];
          if ~isempty(xi)
            m = histc(cP(xi),1:nb);
            p = cumsum(m);
            p = [0; p(:)];
            [~,idx] = sort(cP(xi));
            xi = xi(idx);
            for j = box2ctr{box}
              blocks(j).cxi = [blocks(j).cxi xi(p(j)+1:p(j+1))];
              blocks(j).cprnt = [blocks(j).cprnt pbox*ones(1,m(j))];
            end
          end
        end

        % keep only nonempty centers
        m = histc(rP(rrem),1:nb) + histc(cP(crem),1:nb);
        idx = m > 0;
        ctr = ctr(idx,:);
        blocks = blocks(idx);
        nb = length(blocks);
        for i = 1:nb
          blocks(i).ctr = ctr(i,:);
        end
        p = cumsum(m == 0);
        for box = 1:nbox
          box2ctr{box} = box2ctr{box}(idx(box2ctr{box}));
          box2ctr{box} = box2ctr{box} - p(box2ctr{box})';
        end

        % find neighbors for each center
        proc = false(nb,1);
        for box = 1:nbox
          j = t.nodes(t.lvp(lvl)+box).nbor;
          j = j(j <= t.lvp(lvl));
          for i = box2ctr{box}
            blocks(i).nbr1 = [blocks(i).nbr1 j];
          end
          slf = box2ctr{box};
          nbr = t.nodes(t.lvp(lvl)+box).nbor;
          nbr = nbr(nbr > t.lvp(lvl)) - t.lvp(lvl);
          nbr = unique([box2ctr{[box nbr]}]);
          dx = abs(round(bsxfun(@minus,ctr(slf,1),ctr(nbr,1)')/l));
          dy = abs(round(bsxfun(@minus,ctr(slf,2),ctr(nbr,2)')/l));
          nrx = bsxfun(@le,dx,1);
          nry = bsxfun(@le,dy,1);
          near = nrx & nry;
          for i = 1:length(slf)
            j = slf(i);
            if ~proc(j)
              k = nbr(near(i,:));
              blocks(j).nbr2 = k(k ~= j);
              proc(j) = 1;
            end
          end
        end
      end

      % initialize
      nlvl = nlvl + 1;
      if d == 2
        nb = t.lvp(lvl+1) - t.lvp(lvl);
      else
        nb = length(blocks);
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          t.nodes(i).rxi = [];
          t.nodes(i).cxi = [];
        end
      end
      nrrem1 = sum(rrem);
      ncrem1 = sum(crem);
      nrcrem1 = sum(rcrem);
      nccrem1 = sum(ccrem);
      nblk = pblk(lvl) + nb;
      nz = 0;

      % loop over blocks
      for i = 1:nb
        if d == 2
          j = t.lvp(lvl) + i;
          blk = t.nodes(j);
          rnbr = [t.nodes(blk.nbor).rxi];
          cnbr = [t.nodes(blk.nbor).cxi];
        else
          blk = blocks(i);
          rnbr = [[t.nodes(blk.nbr1).rxi] [blocks(blk.nbr2).rxi]];
          cnbr = [[t.nodes(blk.nbr1).cxi] [blocks(blk.nbr2).cxi]];
        end
        rslf = blk.rxi;
        cslf = blk.cxi;
        nrslf = length(rslf);
        ncslf = length(cslf);

        % find restriction to "skeletonizable" set
        rcslf = find(rcrem(rslf))';
        ccslf = find(ccrem(cslf))';
        rnbr = rnbr(find(rcrem(rnbr)));
        cnbr = cnbr(find(ccrem(cnbr)));

        % compute row proxy interactions and subselect neighbors
        Kpxy = zeros(nrslf,0);
        if lvl > 2
          if isempty(pxyfun)
            cnbr = setdiff(find(ccrem),cslf(ccslf));
          else
            [Kpxy,cnbr] = pxyfun('r',rx,cx,rslf,cnbr,l,blk.ctr);
          end
        end

        % add neighbors with modified interactions
        [~,mod] = find(X(rslf,:));
        mod = unique(mod);
        mod = mod(~ismemb(mod,sort(cslf)));
        cnbr = unique([cnbr(:); mod(:)]);

        % compute interaction matrix
        K1 = full(A(rslf,cnbr));
        [K2,cP] = spget(X,rslf,cnbr,cP);
        K = [K1+K2 Kpxy]';

        % skeletonize rows
        [rsk,~,~] = id(K(:,rcslf),rank_or_tol,0);
        rsk = rcslf(rsk);
        rrd = setdiff(1:nrslf,rsk);
        rT = K(:,rsk)\K(:,rrd);

        % compute column proxy interactions and subselect neighbors
        Kpxy = zeros(0,ncslf);
        if lvl > 2
          if isempty(pxyfun)
            rnbr = setdiff(find(rcrem),rslf(rcslf));
          else
            [Kpxy,cnbr] = pxyfun('c',rx,cx,cslf,rnbr,l,blk.ctr);
          end
        end

        % add neighbors with modified interactions
        [mod,~] = find(X(:,cslf));
        mod = unique(mod);
        mod = mod(~ismemb(mod,sort(rslf)));
        rnbr = unique([rnbr(:); mod(:)]);

        % compute interaction matrix
        K1 = full(A(rnbr,cslf));
        [K2,cP] = spget(X,rnbr,cslf,cP);
        K = [K1+K2; Kpxy];

        % skeletonize columns
        [csk,~,~] = id(K(:,ccslf),rank_or_tol,0);
        csk = ccslf(csk);
        crd = setdiff(1:ncslf,csk);
        cT = K(:,csk)\K(:,crd);

        prrd = rrd;
        pcrd = crd;

        % find good redundant pivots
        % - pivoting moves redundants to skeletons
        % - randomly filter added skeletons to not add too many for compression
        [tmp,cP] = spget(X,rslf,cslf,cP);
        K = full(A(rslf,cslf)) + tmp;
        if lvl > 1
          nrrd = length(rrd);
          ncrd = length(crd);
          if nrrd > ncrd
            nkeep = length(rsk);
            [rsk,rrd,rT] = rdpivot('r',K(rrd,crd),rsk,rrd,rT);
            nadd = nrrd - length(rrd);
            iadd = rsk(end-nadd+1:end);
            idx = randperm(nadd,max(nadd-nkeep,0));
            prrd = [rrd iadd(idx)];
          elseif nrrd < ncrd
            nkeep = length(csk);
            [csk,crd,cT] = rdpivot('c',K(rrd,crd),csk,crd,cT);
            nadd = ncrd - length(crd);
            iadd = csk(end-nadd+1:end);
            idx = randperm(nadd,max(nadd-nkeep,0));
            pcrd = [crd iadd(idx)];
          end
        end

        % update "skeletonizable" set
        rcrem(rslf(prrd)) = 0;
        ccrem(cslf(pcrd)) = 0;

        % restrict to skeletons
        if d == 2
          t.nodes(j).rxi = rslf(rsk);
          t.nodes(j).cxi = cslf(csk);
        else
          for j = rsk
            t.nodes(blk.rprnt(j)).rxi = [t.nodes(blk.rprnt(j)).rxi rslf(j)];
          end
          for j = csk
            t.nodes(blk.cprnt(j)).cxi = [t.nodes(blk.cprnt(j)).cxi cslf(j)];
          end
        end

        % move on if no compression
        if isempty(rrd) && isempty(crd)
          continue
        end
        rrem(rslf(rrd)) = 0;
        crem(cslf(crd)) = 0;

        % compute factors
        [tmp,cP] = spget(X,rslf,cslf,cP);
        K = full(A(rslf,cslf)) + tmp;
        K(rrd,:) = K(rrd,:) - rT'*K(rsk,:);
        K(:,crd) = K(:,crd) - K(:,csk)*cT;
        Krd = K(rrd,crd);
        [nrrd,ncrd] = size(Krd);
        if nrrd > ncrd      % can only happen at root
          [L,U] = qr(Krd,0);
          E = zeros(0,nrrd);
          G = zeros(ncrd,0);
        elseif nrrd < ncrd  % can only happen at root
          [Q,R] = qr(Krd',0);
          L = R';
          U = Q';
          E = zeros(0,nrrd);
          G = zeros(ncrd,0);
        else                % for all non-root
          [L,U] = lu(Krd);
          E = K(rsk,crd)/U;
          G = L\K(rrd,csk);

          % update self-interaction
          S_ = -E*G;
          [I_,J_] = ndgrid(rslf(rsk),cslf(csk));
          m = length(rsk)*length(csk);
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
        end

        % store matrix factors
        n = n + 1;
        while mn < n
          e = cell(mn,1);
          s = struct('rsk',e,'rrd',e,'csk',e,'crd',e,'rT',e,'cT',e, ...
                     'E',e,'F',e,'L',e,'U',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
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
      end
      F.lvp(nlvl+1) = n;

      % update modified entries
      [I_,J_,S_] = find(X);
      idx = rrem(I_) & crem(J_);
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
      X = sparse(I(1:nz),J(1:nz),S(1:nz),M,N);

      % print summary
      if opts.verb
        nrrem2 = sum(rrem);
        ncrem2 = sum(crem);
        nrcrem2 = sum(rcrem);
        nccrem2 = sum(ccrem);
        nblk = pblk(lvl) + t.lvp(lvl+1) - t.lvp(lvl);
        fprintf('%3d-%1d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e (s)\n', ...
                lvl,d,nblk,nrrem1,nrrem2,nrrem1/nblk,nrrem2/nblk,toc)
        fprintf('%5s | %6s | %8d | %8d | %8.2f | %8.2f | %10s (s)\n', ...
                ' ',' ',nrcrem1,nrcrem2,nrcrem1/nblk,nrcrem2/nblk,'')
        fprintf('%5s | %6d | %8d | %8d | %8.2f | %8.2f | %10s (s)\n', ...
                ' ',nblk,ncrem1,ncrem2,ncrem1/nblk,ncrem2/nblk,'')
        fprintf('%5s | %6s | %8d | %8d | %8.2f | %8.2f | %10s (s)\n', ...
                ' ',' ',nccrem1,nccrem2,nccrem1/nblk,nccrem2/nblk,'')
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