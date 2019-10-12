% HIFIE3  Hierarchical interpolative factorization for integral equations in 3D.
%
%    This is essentially the same as HIFIE2 but extended to 3D, compressing
%    first cells to faces, then faces to edges, and finally edges to points.
%
%    This implementation is suited for first-kind integral equations or matrices
%    that otherwise do not have high contrast between the diagonal and off-
%    diagonal elements. For second-kind integral equations, see HIFIE3X.
%
%    Typical complexity for N = SIZE(A,1) = SIZE(A,2): O(N).
%
%    See also HIFIE2, HIFIE2X, HIFIE3X, HIFIE_CHOLMV, HIFIE_CHOLSV, HIFIE_DIAG,
%    HIFIE_LOGDET, HIFIE_MV, HIFIE_SPDIAG, HIFIE_SV.

function F = hifie3(A,x,occ,rank_or_tol,pxyfun,opts)

  % set default parameters
  if nargin < 5, pxyfun = []; end
  if nargin < 6, opts = []; end
  if ~isfield(opts,'lvlmax'), opts.lvlmax = Inf; end
  if ~isfield(opts,'ext'), opts.ext = []; end
  if ~isfield(opts,'skip'), opts.skip = 0; end
  if ~isfield(opts,'symm'), opts.symm = 'n'; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % check inputs
  assert(opts.skip >= 0,'FLAM:hifie3:invalidSkip', ...
         'Skip parameter must be nonnegative.')
  opts.symm = chksymm(opts.symm);
  if opts.symm == 'h' && isoctave()
    warning('FLAM:hifie3:octaveLDL','No LDL decomposition in Octave; using LU.')
    opts.symm = 'n';
  end

  % print header
  if opts.verb
    fprintf([repmat('-',1,71) '\n'])
    fprintf('%5s | %6s | %19s | %19s | %10s\n', ...
            'lvl','nblk','start/end npts','start/end npts/blk','time (s)')
    fprintf([repmat('-',1,71) '\n'])
  end

  % build tree
  N = size(x,2);
  ts = tic;
  t = hypoct(x,occ,opts.lvlmax,opts.ext);
  te = toc(ts);
  if opts.verb, fprintf('%5s | %63.2e\n','t',te); end

  % count nonempty boxes at each level
  pblk = zeros(t.nlvl+1,1);
  for lvl = 1:t.nlvl
    pblk(lvl+1) = pblk(lvl);
    for i = t.lvp(lvl)+1:t.lvp(lvl+1)
      if ~isempty(t.nodes(i).xi), pblk(lvl+1) = pblk(lvl+1) + 1; end
    end
  end

  % initialize
  mn = t.lvp(end);  % maximum capacity for matrix factors
  e = cell(mn,1);
  F = struct('sk',e,'rd',e,'T',e,'L',e,'U',e,'p',e,'E',e,'F',e);
  F = struct('N',N,'nlvl',0,'lvp',zeros(1,2*t.nlvl+1),'factors',F,'symm', ...
             opts.symm);
  nlvl = 0;
  n = 0;
  rem = true(N,1);  % which points remain?
  M = sparse(N,N);  % sparse matrix for Schur complement updates
  nz = 128;         % initial capacity for sparse matrix workspace
  I = zeros(nz,1);
  J = zeros(nz,1);
  V = zeros(nz,1);
  P = zeros(N,1);   % for indexing

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
      ts = tic;

      % dimension reduction
      if d < 3
        if lvl == 1, break; end                     % done if at root
        if lvl > t.nlvl - opts.skip, continue; end  % continue if in skip stage

        % generate face centers
        if d == 2
          ctr = zeros(6*nbox,3);
          box2ctr = cell(nbox,1);
          for i = t.lvp(lvl)+1:t.lvp(lvl+1)  % cells on this level
            j = i - t.lvp(lvl);              % local cell index
            idx = 6*(j-1)+1:6*j;             % face indices
            off = [0 0 -1; 0 -1 0; -1 0 0;   % offset from cell center
                   0 0  1; 0  1 0;  1 0 0];
            ctr(idx,:) = t.nodes(i).ctr + 0.5*l*off;  % face centers
            box2ctr{j} = idx;  % mapping from each cell to its faces
          end

        % generate edge centers
        elseif d == 1
          ctr = zeros(12*nbox,3);
          box2ctr = cell(nbox,1);
          for i = t.lvp(lvl)+1:t.lvp(lvl+1)  % cells on this level
            j = i - t.lvp(lvl);              % local cell index
            idx = 12*(j-1)+1:12*j;           % edge indices
            off = [ 0 -1 -1;  0 -1 1; 0  1 -1; 0 1 1;  % offset from cell center
                   -1  0 -1; -1  0 1; 1  0 -1; 1 0 1;
                   -1 -1  0; -1  1 0; 1 -1  0; 1 1 0];
            ctr(idx,:) = t.nodes(i).ctr + 0.5*l*off;  % edge centers
            box2ctr{j} = idx;  % mapping from each cell to its edges
          end
        end

        % restrict to unique shared centers
        idx = round(2*(ctr - t.nodes(1).ctr)/l);  % displacement from root
        [~,i,j] = unique(idx,'rows');             % unique indices
        p = find(histc(j,1:max(j)) > 1);          % shared indices
        % for edges, further restrict to those shared diagonally across boxes
        if d == 1
          np = length(p);             % number of shared centers
          keep = false(np,1);         % which of those to keep?
          k = repmat(1:nbox,12,1); k = k(:);  % parent box indices
          [~,q] = sort(j); k = k(q);  % sort by mapping to shared centers
          q = [0; find(diff(j(q))); length(j)];  % index array to shared centers
          for ip = 1:np               % loop over shared centers
            c = [t.nodes(t.lvp(lvl)+k(q(p(ip))+1:q(p(ip)+1))).ctr];
            c = reshape(c,3,[])';     % centers of corresponding parent boxes
            dx = c(:,1) - c(:,1)';    % distance between boxes
            dy = c(:,2) - c(:,2)';
            dz = c(:,3) - c(:,3)';
            dist = round((abs(dx) + abs(dy) + abs(dz))/l);  % convert to integer
            if any(dist(:) == 2), keep(ip) = 1; end  % diagonal -> dist = 2
          end
          p = p(keep);
        end
        ctr = ctr(i(p),:);           % remaining centers
        idx = zeros(size(idx,1),1);  % mapping from each index to ...
        idx(p) = 1:length(p);        % ... remaining index or none
        for box = 1:nbox, box2ctr{box} = nonzeros(idx(j(box2ctr{box})))'; end

        % initialize center data structure
        nb = size(ctr,1);
        e = cell(nb,1);
        blocks = struct('ctr',e,'xi',e,'prnt',e,'nbr1',e,'nbr2',e);

        % sort points by centers
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];  % points in this cell
          i = box2ctr{box};                   % associated centers
          dx = x(1,xi) - ctr(i,1);
          dy = x(2,xi) - ctr(i,2);
          dz = x(3,xi) - ctr(i,3);
          dist = sqrt(dx.^2 + dy.^2 + dz.^2);
          [~,near] = max(dist == min(dist));  % nearest center to each point
          P(xi) = box2ctr{box}(near);         % assign points to first nearest
        end
        for box = 1:nbox
          xi = [t.nodes(t.lvp(lvl)+box).xi];
          if isempty(xi), continue; end
          cnt = histc(P(xi),1:nb);              % num of points for each center
          [~,idx] = sort(P(xi)); xi = xi(idx);  % sort points by centers
          p = [0; cumsum(cnt)];  % index array from sorted points to centers
          for i = box2ctr{box}
            blocks(i).xi = [blocks(i).xi xi(p(i)+1:p(i+1))];
            blocks(i).prnt = [blocks(i).prnt (t.lvp(lvl)+box)*ones(1,cnt(i))];
          end
        end

        % restrict to nonempty centers
        cnt = histc(P(rem),1:nb);
        idx = cnt > 0;
        ctr = ctr(idx,:);
        blocks = blocks(idx);
        nb = length(blocks);   % number of nonempty centers
        for i = 1:nb, blocks(i).ctr = ctr(i,:); end
        p = cumsum(cnt == 0);  % how many empty boxes before this one?
        for box = 1:nbox
          box2ctr{box} = box2ctr{box}(idx(box2ctr{box}));  % nonempty only
          box2ctr{box} = box2ctr{box} - p(box2ctr{box})';  % re-index
        end

        % find neighbors for each center
        proc = false(nb,1);
        for box = 1:nbox
          slf = box2ctr{box};
          nbr = t.nodes(t.lvp(lvl)+box).nbor;  % neighbors for this cell
          nbr1 = nbr(nbr <= t.lvp(lvl));       % nbr1 = higher-level cells
          for i = slf, blocks(i).nbr1 = [blocks(i).nbr1 nbr1]; end
          nbr2 = nbr(nbr > t.lvp(lvl)) - t.lvp(lvl);  % nbr2 = same-level ...
          nbr2 = unique([slf box2ctr{nbr2}]);         %        ... centers
          dx = abs(round((ctr(slf,1) - ctr(nbr2,1)')/l))';
          dy = abs(round((ctr(slf,2) - ctr(nbr2,2)')/l))';
          dz = abs(round((ctr(slf,3) - ctr(nbr2,3)')/l))';
          near = dx <= 1 & dy <= 1 & dz <= 1;  % check if actually neighbors
          for i = 1:length(slf)
            j = slf(i);
            if proc(j), continue; end    % already processed
            k = nbr2(near(:,i));
            blocks(j).nbr2 = k(k ~= j);  % don't include self
            proc(j) = true;
          end
        end
      end

      % initialize for compression/factorization
      nlvl = nlvl + 1;
      if d == 3, nb = t.lvp(lvl+1) - t.lvp(lvl);  % use cells
      else,      nb = length(blocks);             % use centers
        for i = t.lvp(lvl)+1:t.lvp(lvl+1), t.nodes(i).xi = []; end
      end
      nrem1 = sum(rem);  % remaining points at start
      nz = 0;

      % loop over blocks
      for i = 1:nb
        if d == 3
          j = t.lvp(lvl) + i;  % global tree index
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
          if isempty(pxyfun), nbr = setdiff(find(rem),slf);
          else, [Kpxy,nbr] = pxyfun(x,slf,nbr,l,blk.ctr);
          end
        end

        % add neighbors with modified interactions
        [nbr_mod,~] = find(M(:,slf));
        nbr_mod = unique(nbr_mod);
        nbr_mod = nbr_mod(~ismemb(nbr_mod,sslf));
        nbr = unique([nbr(:); nbr_mod(:)]);

        % compress off-diagonal block
        K1 = full(A(nbr,slf));
        if opts.symm == 'n', K1 = [K1; full(A(slf,nbr))']; end
        K2 = spget(M,nbr,slf);
        if opts.symm == 'n', K2 = [K2; spget(M,slf,nbr)']; end
        K = [K1 + K2; Kpxy];
        [sk,rd,T] = id(K,rank_or_tol,0);  % no randomization for better ranks

        % restrict to skeletons for next level
        if d == 3
          t.nodes(j).xi = slf(sk);
        else
          for j = sk
            t.nodes(blk.prnt(j)).xi = [t.nodes(blk.prnt(j)).xi slf(j)];
          end
        end

        % move on if no compression
        if isempty(rd), continue; end
        rem(slf(rd)) = 0;

        % compute factors
        K = full(A(slf,slf)) + spget(M,slf,slf);
        if opts.symm == 's', K(rd,:) = K(rd,:) - T.'*K(sk,:);
        else,                K(rd,:) = K(rd,:) - T' *K(sk,:);
        end
        K(:,rd) = K(:,rd) - K(:,sk)*T;
        if opts.symm == 'n' || opts.symm == 's'
          [L,U,p] = lu(K(rd,rd),'vector');
          E = K(sk,rd)/U;
          G = L\K(rd(p),sk);
        elseif opts.symm == 'h'
          [L,U,p] = ldl(K(rd,rd),'vector');
          U = sparse(U);
          E = (K(sk,rd(p))/L')/U.';
          G = [];
        elseif opts.symm == 'p'
          L = chol(K(rd,rd),'lower');
          E = K(sk,rd)/L';
          U = []; p = []; G = [];
        end

        % update self-interaction
        if     opts.symm == 'h', X = -E*(U*E');
        elseif opts.symm == 'p', X = -E*E';
        else,                    X = -E*G;
        end
        [I_,J_] = ndgrid(slf(sk));
        [I,J,V,nz] = sppush3(I,J,V,nz,I_,J_,X);

        % store matrix factors
        n = n + 1;
        if mn < n
          e = cell(mn,1);
          s = struct('sk',e,'rd',e,'T',e,'L',e,'U',e,'p',e,'E',e,'F',e);
          F.factors = [F.factors; s];
          mn = 2*mn;
        end
        F.factors(n).sk = slf(sk);
        F.factors(n).rd = slf(rd);
        F.factors(n).T = T;
        F.factors(n).L = L;
        F.factors(n).U = U;
        F.factors(n).p = p;
        F.factors(n).E = E;
        F.factors(n).F = G;
      end
      F.lvp(nlvl+1) = n;

      % update modified entries
      [I_,J_,V_] = find(M);
      idx = rem(I_) & rem(J_);
      [I,J,V,nz] = sppush3(I,J,V,nz,I_(idx),J_(idx),V_(idx));
      M = sparse(I(1:nz),J(1:nz),V(1:nz),N,N);
      te = toc(ts);

      % print summary
      if opts.verb
        nrem2 = sum(rem);       % remaining points at end
        nblk = pblk(lvl) + nb;  % nonempty up to this level
        fprintf('%3d-%1d | %6d | %8d | %8d | %8.2f | %8.2f | %10.2e\n', ...
                lvl,d,nblk,nrem1,nrem2,nrem1/nblk,nrem2/nblk,te)
      end
    end
  end

  % finish
  F.nlvl = nlvl;
  F.lvp = F.lvp(1:nlvl+1);
  F.factors = F.factors(1:n);
  if opts.verb, fprintf([repmat('-',1,71) '\n']); end
end