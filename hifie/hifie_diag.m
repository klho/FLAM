% HIFIE_DIAG  Extract diagonal using hierarchical interpolative factorization
%             for integral equations via matrix unfolding.
%
%    This algorithm exploits the fact that only a small subset of all matrix
%    entries need to be reconstructed from the top-level skeletons in order to
%    compute the diagonal.
%
%    Typical complexity: same as factorization. However, the constant can be
%    quite large due to the coupling of points across cell boundaries so that
%    this may not be too practical; see HIFIE_SPDIAG for an alternative.
%
%    D = HIFIE_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = HIFIE_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    D = HIFIE_DIAG(F,DINV,OPTS) also passes various options to the algorithm.
%    Valid options include:
%
%      - VERB: display status info if VERB = 1 (default: VERB = 0). This prints
%              to screen a table tracking extraction statistics through
%              factorization level (i.e., tree leaves are at level 1). Special
%              levels: 'A', for determining all required entries to compute.
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_SPDIAG.

function D = hifie_diag(F,dinv,opts)

  % set default parameters
  if nargin < 2, dinv = 0; end
  if nargin < 3, opts = []; end
  if ~isfield(opts,'verb'), opts.verb = 0; end

  % print header
  if opts.verb
    fprintf([repmat('-',1,31) '\n'])
    fprintf('%3s | %12s | %10s\n','lvl','nnz kept','time (s)')
    fprintf([repmat('-',1,31) '\n'])
  end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  rem = true(N,1);  % which points remain?
  nz = N;           % initial capacity for sparse matrix workspace
  I = zeros(nz,1);
  J = zeros(nz,1);

  % find required entries at each level
  ts = tic;
  keep = cell(nlvl,1);  % entries to keep after unfolding at each level
  keep{1} = sparse(1:N,1:N,true(N,1),N,N);  % at leaf, just need diagonal
  for lvl = 1:nlvl-1  % current level is lvl+1
    nz = 0;

    % eliminate redundant indices
    rem([F.factors(F.lvp(lvl)+1:F.lvp(lvl+1)).rd]) = false;

    % keep entries needed directly by previous level
    [I_,J_] = find(keep{lvl});
    idx = rem(I_) & rem(J_);
    [I,J,nz] = sppush2(I,J,nz,I_(idx),J_(idx));

    % store transpose for fast row access
    keeptrans = keep{lvl}';

    % loop over nodes at previous level
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      f = F.factors(i);

      % keep skeleton-skeleton entries
      sk = f.sk;
      [I_,J_] = ndgrid(sk);
      [I,J,nz] = sppush2(I,J,nz,I_,J_);

      % keep skeleton-external entries
      if lvl == 1, continue; end
      rd = f.rd;
      [I_,~] = find(keep{lvl}(:,sk));
      [J_,~] = find(keeptrans(:,sk));
      I_ = unique([I_(rem(I_)); J_(rem(J_))]);
      ex = I_(~ismemb(I_,sort([rd sk])));
      [I_,J_] = ndgrid(ex,sk);
      I__ = I_(:);
      J__ = J_(:);
      I_ = [I__; J__];
      J_ = [J__; I__];
      [I,J,nz] = sppush2(I,J,nz,I_,J_);
    end

    % construct requirement matrix
    idx = 1:nz;
    if F.symm ~= 'n', idx = find(I(idx) >= J(idx)); end
    if isoctave(), keep{lvl+1} = sparse(I(idx),J(idx),true(size(idx)),N,N);
    else, keep{lvl+1} = logical(sparse(I(idx),J(idx),ones(size(idx)),N,N));
    end
  end

  % allocate storage for all required entries in sparse column array form for
  % efficient per-node updating (note: technically only needed if DINV = 1)
  keep_ = keep{1};
  for lvl = 1:nlvl-1, keep_ = keep_ | keep{lvl+1}; end
  [~,J] = find(keep_);
  nz = diff([0; find(diff(J)); length(J)]);  % number of nonzeros in each column
  M = cell(N,1);                             % successively unfolded matrix ...
  for i = 1:N, M{i} = spalloc(N,1,nz(i)); end  % ... stored column by column
  % use of this data structure has a large prefactor cost but appears essential
  % for achieving the correct asymptotic scaling
  t = toc(ts);

  % print summary
  if opts.verb, fprintf('%3s | %12d | %10.2e\n','a',nnz(keep_),t); end

  % unfold factorization
  for lvl = nlvl:-1:1  % loop from top-down
    ts = tic;
    keeptrans = keep{lvl}';  % store transpose for fast row access

    % loop over nodes
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      f = F.factors(i);
      sk = f.sk; rd = f.rd;

      T = f.T;
      L = f.L;
      p = f.p;
      E = f.E;
      if F.symm == 'n' || F.symm == 's'
        U = f.U;
        G = f.F;
      else
        U = f.L';
        G = f.E';
      end

      % find external interactions
      [I_,~] = find(keep{lvl}(:,sk));
      [J_,~] = find(keeptrans(:,sk));
      I_ = unique([I_; J_]);
      ex = I_(~ismemb(I_,sort([rd sk])))';

      % unfold local factorization
      nrd = length(rd);
      nsk = length(sk);
      nex = length(ex);
      ird = 1:nrd;
      isk = nrd+(1:nsk);
      iex = nrd+nsk+(1:nex);
      X = zeros(nrd+nsk+nex);
      % redundant part
      if F.symm == 'h'
        if dinv, X(ird,ird) = inv(f.U);
        else,    X(ird,ird) =     f.U ;
        end
      else,      X(ird,ird) = eye(nrd);
      end
      % skeleton and external part
      Xse = spgetv(M,[sk ex],[sk ex]);
      Xse(1:nsk,1:nsk) = spsymm(Xse(1:nsk,1:nsk),F.symm);
      ise = [isk iex];
      X(ise,ise) = Xse;
      [X(iex,isk),X(isk,iex)] = spsymm2(X(iex,isk),X(isk,iex),F.symm);
      % undo elimination and sparsification
      if dinv
        X(:,ird) = (X(:,ird) - X(:,isk)*E)/L;
        X(ird,:) = U\(X(ird,:) - G*X(isk,:));
        if ~isempty(p)
          X(:,ird(p)) = X(:,ird);
          if F.symm == 'h', X(ird(p),:) = X(ird,:); end
        end
        if F.symm == 's', X(:,isk) = X(:,isk) - X(:,ird)*T.';
        else,             X(:,isk) = X(:,isk) - X(:,ird)*T' ;
        end
        X(isk,:) = X(isk,:) - T*X(ird,:);
      else
        X(:,isk) = X(:,isk) + X(:,ird)*G;
        X(isk,:) = X(isk,:) + E*X(ird,:);
        X(:,ird) = X(:,ird)*U;
        X(ird,:) = L*X(ird,:);
        if ~isempty(p)
          if F.symm == 'h', X(:,ird(p)) = X(:,ird); end
          X(ird(p),:) = X(ird,:);
        end
        X(:,ird) = X(:,ird) + X(:,isk)*T;
        if F.symm == 's', X(ird,:) = X(ird,:) + T.'*X(isk,:);
        else,             X(ird,:) = X(ird,:) + T' *X(isk,:);
        end
      end
      X(ise,ise) = X(ise,ise) - Xse;  % to be stored as update

      % store update to global sparse matrix
      rse = [rd sk ex];
      X = X .* spget(keep{lvl},rse,rse);
      M = spaddv(M,rse,rse,X);
    end
    t = toc(ts);

    % print summary
    if opts.verb, fprintf('%3d | %12d | %10.2e\n',lvl,nnz(keep{lvl}),t); end
  end

  % finish
  D = zeros(N,1);
  for i = 1:N, D(i) = M{i}(i); end
  if opts.verb, fprintf([repmat('-',1,31) '\n']); end
end