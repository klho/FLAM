% HYPOCT  Build hyperoctree.
%
%    A hyperoctree is a direct generalization of the octree in 3D to arbitrary
%    dimension, wherein a given node is bisected in up to all dimensions on
%    refining from one level to the next. This implementation constructs an
%    adaptive hyperoctree on approximately hypercube nodes (i.e., once
%    sufficiently refined, all nodes have bounded aspect ratio <= SQRT(2)). In
%    addition to the usual parent-child relationships, it also computes the
%    neighbors of each node.
%
%    T = HYPOCT(X,OCC) builds a hyperoctree T over a set of points X such that
%    each leaf node contains at most OCC points. The point array X should have
%    size [D N], where D is the dimension of the space and N is the number of
%    points. The output tree T is structured as follows:
%
%        NLVL  - tree depth
%        LVP   - tree level pointer array
%        L     - tree node size
%        NODES - tree node data array
%
%    Each element of NODES further has fields:
%
%        CTR  - node center
%        XI   - node point indices
%        PRNT - node parent
%        CHLD - node children
%        NBOR - node neighbors
%
%    Some examples of how to access the tree data are given below:
%
%      - The nodes on level 1 <= L <= NLVL are T.NODES(T.LVP(L)+1:T.LVP(L+1)).
%      - The size of each node on level LVL is T.L(:,LVL).
%      - The points in node index I are X(:,T.NODES(I).XI).
%      - The parent of node index I is T.NODES(T.NODES(I).PRNT).
%      - The children of node index I are [T.NODES(T.NODES(I).CHLD)].
%      - The neighbors of node index I are [T.NODES(T.NODES(I).NBOR)].
%
%    T = HYPOCT(X,OCC,LVLMAX) builds a hyperoctree to a maximum depth LVLMAX
%    (default: LVLMAX = INF). If this limit is hit, then the leaf nodes may not
%    necessarily obey the occupancy constraint set by OCC.
%
%    T = HYPOCT(X,OCC,LVLMAX,EXT) sets the root node extent to
%    [EXT(D,1) EXT(D,2)] along dimension D. If EXT is empty (default), then the
%    root extent is calculated from the data.
%
%    References:
%
%      H. Samet. The quadtree and related hierarchical data structures. ACM
%        Comput. Surv. 16 (2): 187-260, 1984.

function T = hypoct(x,occ,lvlmax,ext)

  % set default parameters
  if nargin < 3 || isempty(lvlmax), lvlmax = Inf; end
  if nargin < 4, ext = []; end

  % check inputs
  assert(occ >= 0,'FLAM:hypoct:invalidOcc', ...
         'Leaf occupancy must be nonnegative.')
  assert(lvlmax >= 1,'FLAM:hypoct:invalidLvlmax', ...
         'Maximum tree depth must be at least 1.')

  % initialize root node
  [d,n] = size(x);
  if isempty(ext), ext = [min(x,[],2) max(x,[],2)]; end
  l = ext(:,2) - ext(:,1);
  ctr = 0.5*(ext(:,1) + ext(:,2));
  s = struct('ctr',ctr,'xi',1:n,'prnt',[],'chld',[],'nbor',[]);
  T = struct('nlvl',1,'lvp',[0 1],'l',l,'nodes',s);
  nlvl = 1; nbox = 1;  % number of levels and boxes
  mlvl = 1; mbox = 1;  % total allocated capacity

  % main loop
  while 1

    % terminate if at maximum depth
    if nlvl >= lvlmax, break; end

    % initialize level
    nbox_ = nbox;
    ldiv = l >= max(l)/sqrt(2);  % bisect only "long" sides
    l(ldiv) = 0.5*l(ldiv);

    % loop over all boxes at current level
    for prnt = T.lvp(nlvl)+1:T.lvp(nlvl+1)
      xi = T.nodes(prnt).xi;
      xn = length(xi);

      % subdivide box if it contains too many points
      if xn > occ
        ctr = T.nodes(prnt).ctr;
        idx = ldiv.*(x(:,xi) > ctr);  % which side of center in each dim?
        idx = 2.^((1:d) - 1)*idx;     % convert d-vector to integer
        uidx = unique(idx);           % nonempty child boxes

        % exponentially increase capacity as needed
        nbox_new = nbox + length(uidx);
        if mbox < nbox_new
          while mbox < nbox_new, mbox = 2*mbox; end
          e = cell(mbox-length(T.nodes),1);
          s = struct('ctr',e,'xi',e,'prnt',e,'chld',e,'nbor',e);
          T.nodes = [T.nodes; s];
        end

        % store child box data
        for i = uidx
          nbox = nbox + 1;
          s = struct( 'ctr', ctr + l.*ldiv.*(bitget(i,1:d) - 0.5)', ...
                       'xi', xi(idx == i),                           ...
                     'prnt', prnt,                                   ...
                     'chld', [],                                     ...
                     'nbor', []);
          T.nodes(nbox) = s;
          T.nodes(prnt).chld = [T.nodes(prnt).chld nbox];
        end
        T.nodes(prnt).xi = [];  % points moved to children; clear parent
      end
    end

    % terminate if no new boxes added
    if nbox <= nbox_, break; end

    % update for new tree level
    nlvl = nlvl + 1;
    T.nlvl = nlvl;
    if mlvl < nlvl
      T.lvp = [T.lvp zeros(1,mlvl)];
      T.l   = [T.l   zeros(d,mlvl)];
      mlvl  = 2*mlvl;
    end
    T.lvp(nlvl+1) = nbox;
    T.l(:,nlvl) = l;
  end

  % shrink overallocated arrays
  T.lvp = T.lvp(1:nlvl+1);
  T.l   = T.l(:,1:nlvl);
  T.nodes = T.nodes(1:nbox);

  % find neighbors of each box
  for lvl = 2:nlvl  % root has no neighbors
    l = T.l(:,lvl);
    for i = T.lvp(lvl)+1:T.lvp(lvl+1)
      ctr = T.nodes(i).ctr;
      prnt = T.nodes(i).prnt;

      % add all non-self children of parent
      j = T.nodes(prnt).chld;
      T.nodes(i).nbor = j(j ~= i);

      % add coarser parent-neighbors if adjacent
      for j = T.nodes(prnt).nbor
        if ~isempty(T.nodes(j).xi)
          jctr = T.nodes(j).ctr;
          jlvl = find(j > T.lvp,1,'last');
          jl = T.l(:,jlvl);
          dist = round((abs(ctr - jctr) - 0.5*(l + jl))./l);
          if max(dist) <= 0, T.nodes(i).nbor = [T.nodes(i).nbor j]; end
        end
      end

      % add children of parent-neighbors if adjacent
      idx = [T.nodes(T.nodes(prnt).nbor).chld];
      if ~isempty(idx)
        dist = round(abs(T.nodes(i).ctr - [T.nodes(idx).ctr])./l);
        j = idx(max(dist,[],1) <= 1);
        if ~isempty(j), T.nodes(i).nbor = [T.nodes(i).nbor j]; end
      end
    end
  end
end