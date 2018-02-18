% HIFIE3X  Hierarchical interpolative factorization for integral equations in 3D
%          with accuracy optimizations for second-kind integral equations.
%
%    F = HIFIE3X(A,X,OCC,RANK_OR_TOL,PXYFUN) produces a factorization F of the
%    interaction matrix A on the points X using tree occupancy parameter OCC,
%    local precision parameter RANK_OR_TOL, and proxy function PXYFUN to capture
%    the far field. This is a function of the form
%
%      [KPXY,NBR] = PXYFUN(X,SLF,NBR,L,CTR)
%
%    that is called for every block, where
%
%      - KPXY: interaction matrix against artificial proxy points
%      - NBR:  block neighbor indices (can be modified)
%      - X:    input points
%      - SLF:  block indices
%      - L:    block size
%      - CTR:  block center
%
%    See the examples for further details. If PXYFUN is not provided or empty
%    (default), then the code uses the naive global compression scheme.
%
%    F = HIFIE3X(A,X,OCC,RANK_OR_TOL,PXYFUN,OPTS) also passes various options to
%    the algorithm. Valid options include:
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
%      K.L. Ho, L. Ying. Hierarchical interpolative factorization for elliptic
%        operators: integral equations. Comm. Pure Appl. Math. 69 (7):
%        1314-1353, 2016.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV, HYPOCT, ID.

function F = hifie3x(A,x,occ,rank_or_tol,pxyfun,opts)
  start = tic;

  % set default parameters
  if nargin < 5
    pxyfun = [];
  end
  if nargin < 6
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
  if ~isfield(opts,'symm')
    opts.symm = 'n';
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % check inputs
  assert(opts.skip >= 0,'FLAM:hifie3x:negativeSkip', ...
         'Skip parameter must be nonnegative.')
  assert(strcmpi(opts.symm,'n') || strcmpi(opts.symm,'s') || ...
         strcmpi(opts.symm,'h') || strcmpi(opts.symm,'p'), ...
         'FLAM:hifie3x:invalidSymm', ...
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
    fprintf(' %3s  | %63.2e (s)\n','-',toc)
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
  F = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
  F = struct('N',N,'nlvl',0,'lvp',zeros(1,2*t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = true(N,1);
  mnz = 128;
  M = sparse(N,N);
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  S = zeros(mnz,1);
  P = zeros(N,1);

  % loop over tree levels
  for lvl = t.nlvl:-1:1
    l = t.lrt/2^(lvl - 1);
    nbox = t.lvp(lvl+1) - t.lvp(lvl);

    % pull up skeletons from children
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
        t.nodes(i).xi = [t.nodes(i).xi [t.nodes(t.nodes(i).chld).xi]];
    end

    % loop over dimensions
    for d = [3 2 1]
      tic

      % dimension reduction
      if d < 3

        % continue if in skip stage
        if lvl > t.nlvl - opts.skip
          continue
        end

        % generate face centers
        if d == 2
          ctr = zeros(6*nbox,3);
          box2ctr = cell(nbox,1);
          for i = t.lvp(lvl)+1:t.lvp(lvl+1)
            j = i - t.lvp(lvl);
            idx = 6*(j-1)+1:6*j;
            off = [0 0 -1; 0 -1 0; -1 0 0; 0 0 1; 0 1 0; 1 0 0];
            ctr(idx,:) = bsxfun(@plus,t.nodes(i).ctr,0.5*l*off);
            box2ctr{j} = idx;
          end

        % generate edge centers
        elseif d == 1
          ctr = zeros(12*nbox,3);
          box2ctr = cell(nbox,1);
          ctr2box = zeros(12*nbox,1);
          for i = t.lvp(lvl)+1:t.lvp(lvl+1)
            j = i - t.lvp(lvl);
            idx = 12*(j-1)+1:12*j;
            off = [ 0 -1 -1;  0 -1 1; 0  1 -1; 0 1 1;
                   -1  0 -1; -1  0 1; 1  0 -1; 1 0 1;
                   -1 -1  0; -1  1 0; 1 -1  0; 1 1 0];
            ctr(idx,:) = bsxfun(@plus,t.nodes(i).ctr,0.5*l*off);
            box2ctr{j} = idx;
            ctr2box(idx) = i;
          end
        end

        % find unique shared centers
        idx = bsxfun(@minus,ctr,t.nodes(1).ctr);
        idx = round(2*idx/l);
        [~,i,j] = unique(idx,'rows');
        idx(:) = 0;
        p = find(histc(j,1:max(j)) > 1);

        % for edges, keep only if shared "diagonally" across boxes
        if d == 1
          np = length(p);
          keep = false(np,1);
          for k = 1:np
            c = i(p(k));
            box = ctr2box(c);
            nbr = t.nodes(box).nbor;
            nbr = nbr(nbr > t.lvp(lvl));
            box_ctr = t.nodes(box).ctr;
            nbr_ctr = reshape([t.nodes(nbr).ctr],3,[])';
            dist_ctr = round(2*(box_ctr - ctr(c,:))/l);
            dist_nbr = round(bsxfun(@minus,box_ctr,nbr_ctr)/l);
            if any(all(bsxfun(@eq,dist_nbr,dist_ctr),2))
              keep(k) = 1;
            end
          end
          p = p(keep);
        end
        i = i(p);
        idx(p) = 1:length(p);
        ctr = ctr(i,:);
        for box = 1:nbox
          box2ctr{box} = nonzeros(idx(j(box2ctr{box})))';
        end

        % initialize
        nb = size(ctr,1);
        e = cell(nb,1);
        blocks = struct('ctr',e,'xi',e,'prnt',e,'nbr1',e,'nbr2',e);

        % sort points by centers
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          i = box2ctr{box};
          dx = bsxfun(@minus,x(1,xi),ctr(i,1));
          dy = bsxfun(@minus,x(2,xi),ctr(i,2));
          dz = bsxfun(@minus,x(3,xi),ctr(i,3));
          dist = sqrt(dx.^2 + dy.^2 + dz.^2);
          near = bsxfun(@eq,dist,min(dist,[],1));
          for i = 1:length(xi)
            P(xi(i)) = box2ctr{box}(find(near(:,i),1));
          end
        end
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          if ~isempty(xi)
            m = histc(P(xi),1:nb);
            p = cumsum(m);
            p = [0; p(:)];
            [~,idx] = sort(P(xi));
            xi = xi(idx);
            for j = box2ctr{box}
              blocks(j).xi = [blocks(j).xi xi(p(j)+1:p(j+1))];
              blocks(j).prnt = [blocks(j).prnt (t.lvp(lvl)+box)*ones(1,m(j))];
            end
          end
        end

        % keep only nonempty centers
        m = histc(P(rem),1:nb);
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
          dz = abs(round(bsxfun(@minus,ctr(slf,3),ctr(nbr,3)')/l));
          nrx = bsxfun(@le,dx,1);
          nry = bsxfun(@le,dy,1);
          nrz = bsxfun(@le,dz,1);
          near = nrx & nry & nrz;
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
      if d == 3
        nb = t.lvp(lvl+1) - t.lvp(lvl);
      else
        nb = length(blocks);
        for i = t.lvp(lvl)+1:t.lvp(lvl+1)
          t.nodes(i).xi = [];
        end
      end
      F.lvp(nlvl+1) = F.lvp(nlvl) + nb;
      nblk = pblk(lvl) + nb;
      nrem1 = sum(rem);
      nz = 0;

      % loop over blocks
      for i = 1:nb
        if d == 3
          j = t.lvp(lvl) + i;
          blk = t.nodes(j);
          nbr = [t.nodes(blk.nbor).xi];
        else
          blk = blocks(i);
          nbr = [[t.nodes(blk.nbr1).xi] [blocks(blk.nbr2).xi]];
        end
        slf = blk.xi;
        nslf = length(slf);
        sslf = sort(slf);

        % compute proxy interactions and subselect neighbors
        Kpxy = zeros(0,nslf);
        if lvl > 2
          if isempty(pxyfun)
            nbr = setdiff(find(rem),slf);
          else
            [Kpxy,nbr] = pxyfun(x,slf,nbr,l,blk.ctr);
          end
        end

        % add neighbors with modified interactions
        [mod,~] = find(M(:,slf));
        mod = unique(mod);
        mod = mod(~ismemb(mod,sslf));
        nbr = unique([nbr(:); mod(:)]);
        nnbr = length(nbr);
        snbr = sort(nbr);

        % compute interaction matrix
        K1 = full(A(nbr,slf));
        if strcmpi(opts.symm,'n')
          K1 = [K1; full(A(slf,nbr))'];
        end
        [K2,P] = spget(M,nbr,slf,P);
        if strcmpi(opts.symm,'n')
          [tmp,P] = spget(M,slf,nbr,P);
          K2 = [K2; tmp'];
        end
        K = [K1 + K2; Kpxy];

        % scale compression tolerance
        ratio = 1;
        if rank_or_tol < 1
          nrm1 = snorm(nslf,@(x)(K1*x),@(x)(K1'*x));
          if nnz(K2) > 0
            nrm2 = snorm(nslf,@(x)(K2*x),@(x)(K2'*x));
            ratio = min(1,nrm1/nrm2);
          end
        end

        % partition by sparsity structure of modified interactions
        K2 = double(K2 ~= 0);
        K2 = K2(logical(sum(K2,2)),:);
        s = sum(K2);
        if sum(s) == 0
          grp = {1:nslf};
          ngrp = 1;
        else
          R = K2'*K2;
          s = bsxfun(@max,s',s);
          Krem = true(nslf,1);
          grp = cell(nslf,1);
          ngrp = 0;
          for k = 1:nslf
            if Krem(k)
              idx = find(R(:,k) == s(:,k) & Krem);
              if any(idx)
                ngrp = ngrp + 1;
                grp{ngrp} = idx;
                Krem(idx) = 0;
              end
            end
          end
        end
        grp = grp(1:ngrp);

        % skeletonize by partition
        sk_ = cell(ngrp,1);
        rd_ = cell(ngrp,1);
        T_ = cell(ngrp,1);
        psk = zeros(ngrp,1);
        prd = zeros(ngrp,1);
        for k = 1:ngrp
          K_ = K(:,grp{k});
          Kpxy_ = Kpxy(:,grp{k});
          [sk_{k},rd_{k},T_{k}] = id([K_; Kpxy_],ratio*rank_or_tol,0);
          psk(k) = length(sk_{k});
          prd(k) = length(rd_{k});
        end

        % reassemble skeletonization
        psk = [0; cumsum(psk(:))];
        prd = [0; cumsum(prd(:))];
        sk = zeros(1,psk(end));
        rd = zeros(1,prd(end));
        T = zeros(psk(end),prd(end));
        for k = 1:ngrp
          sk(psk(k)+1:psk(k+1)) = grp{k}(sk_{k});
          rd(prd(k)+1:prd(k+1)) = grp{k}(rd_{k});
          T(psk(k)+1:psk(k+1),prd(k)+1:prd(k+1)) = T_{k};
        end

        % restrict to skeletons
        if d == 3
          t.nodes(j).xi = slf(sk);
        else
          for j = sk
            t.nodes(blk.prnt(j)).xi = [t.nodes(blk.prnt(j)).xi slf(j)];
          end
        end

        % move on if no compression
        if isempty(rd)
          continue
        end
        rem(slf(rd)) = 0;

        % compute factors
        [tmp,P] = spget(M,slf,slf,P);
        K = full(A(slf,slf)) + tmp;
        if strcmpi(opts.symm,'s')
          K(rd,:) = K(rd,:) - T.'*K(sk,:);
        else
          K(rd,:) = K(rd,:) - T'*K(sk,:);
        end
        K(:,rd) = K(:,rd) - K(:,sk)*T;
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
        while mn < n
          e = cell(mn,1);
          s = struct('sk',e,'rd',e,'T',e,'E',e,'F',e,'L',e,'U',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
        F.factors(n).sk = slf(sk);
        F.factors(n).rd = slf(rd);
        F.factors(n).T = T;
        F.factors(n).E = E;
        F.factors(n).F = G;
        F.factors(n).L = L;
        F.factors(n).U = U;
      end
      F.lvp(nlvl+1) = n;

      % update modified entries
      [I_,J_,S_] = find(M);
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
      M = sparse(I(1:nz),J(1:nz),S(1:nz),N,N);

      % print summary
      if opts.verb
        nrem2 = sum(rem);
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
  F.factors = F.factors(1:n);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end
end