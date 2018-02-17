% MF_DIAG  Extract diagonal using multifrontral factorization via matrix
%          unfolding.
%
%    D = MF_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = MF_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    D = MF_DIAG(F,DINV,OPTS) also passes various options to the algorithm.
%    Valid options include:
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    References:
%
%      S. Li, S. Ahmed, G. Klimeck, E. Darve. Computing entries of the inverse
%        of a sparse matrix using the FIND algorithm. J. Comput Phys. 227:
%        9408-9427, 2008.
%
%      L. Lin, J. Lu, L. Ying, R. Car, W. E. Fast algorithm for extracting the
%        diagonal of the inverse matrix with application to the electronic
%        structure analysis of metallic systems. Commun. Math. Sci. 7 (3):
%        755-777, 2009.
%
%    See also MF2, MF3, MF_SPDIAG, MFX.

function D = mf_diag(F,dinv,opts)
  start = tic;

  % set default parameters
  if nargin < 2
    dinv = 0;
  end
  if nargin < 3
    opts = [];
  end
  if ~isfield(opts,'verb')
    opts.verb = 0;
  end

  % initialize
  N = F.N;
  nlvl = F.nlvl;
  rem = true(N,1);
  mnz = 128;
  I = zeros(mnz,1);
  J = zeros(mnz,1);
  P = zeros(N,1);

  % get required entries at each level
  tic
  req = cell(nlvl,1);
  req{1} = sparse(1:N,1:N,true(N,1),N,N);
  for lvl = 1:nlvl-1
    nz = 0;

    % eliminate redundant DOFs
    rem([F.factors(F.lvp(lvl)+1:F.lvp(lvl+1)).rd]) = 0;

    % store previously computed entries
    [I_,J_] = find(req{lvl});
    idx = rem(I_) & rem(J_);
    I_ = I_(idx);
    J_ = J_(idx);
    m = numel(I_);
    while mnz < nz + m
      e = zeros(mnz,1);
      I = [I; e];
      J = [J; e];
      mnz = 2*mnz;
    end
    I(nz+1:nz+m) = I_;
    J(nz+1:nz+m) = J_;
    nz = nz + m;

    % loop over nodes at current level
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)

      % require skeleton entries
      sk = F.factors(i).sk;
      [I_,J_] = ndgrid(sk);
      m = numel(I_);
      while mnz < nz + m
        e = zeros(mnz,1);
        I = [I; e];
        J = [J; e];
        mnz = 2*mnz;
      end
      I(nz+1:nz+m) = I_;
      J(nz+1:nz+m) = J_;
      nz = nz + m;
    end

    % construct requirement matrix
    idx = 1:nz;
    if strcmpi(F.symm,'s') || strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      idx = find(I(idx) >= J(idx));
    end
    req{lvl+1} = logical(sparse(I(idx),J(idx),ones(size(idx)),N,N));
  end

  % print summary
  if opts.verb
    req_ = req{1};
    for lvl = 1:nlvl-1
      req_ = req_ | req{lvl+1};
    end
    fprintf([repmat('-',1,80) '\n'])
    fprintf('%3s | %12d | %25.2e (s)\n','-',nnz(req_),toc)
  end

  % unfold factorization
  S = zeros(mnz,1);
  M = sparse(N,N);
  for lvl = nlvl:-1:1
    tic
    [I_,J_,S_] = find(M);
    nz = length(S_);
    I(1:nz) = I_;
    J(1:nz) = J_;
    S(1:nz) = S_;

    % loop over nodes
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      L = F.factors(i).L;
      E = F.factors(i).E;
      if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
        G = F.factors(i).F;
        U = F.factors(i).U;
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        G = F.factors(i).E';
        U = F.factors(i).L';
      end

      % compute local matrix
      nrd = length(rd);
      nsk = length(sk);
      ird = 1:nrd;
      isk = nrd+1:nrd+nsk;
      D = zeros(nrd+nsk);
      if strcmpi(F.symm,'h')
        if dinv
          D(ird,ird) = inv(F.factors(i).U);
        else
          D(ird,ird) = F.factors(i).U;
        end
      else
        D(ird,ird) = eye(nrd);
      end
      Dsk = spget_sk;
      D(isk,isk) = Dsk;
      if dinv
        D(:,ird) = (D(:,ird) - D(:,isk)*E)/L;
        D(ird,:) = U\(D(ird,:) - G*D(isk,:));
      else
        D(:,isk) = D(:,isk) + D(:,ird)*G;
        D(isk,:) = D(isk,:) + E*D(ird,:);
        D(:,ird) = D(:,ird)*U;
        D(ird,:) = L*D(ird,:);
      end
      D(isk,isk) = D(isk,isk) - Dsk;

      % store update to global sparse matrix
      [I_,J_] = ndgrid([rd sk]);
      S_ = D(:);
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
    end

    % keep only remaining entries
    [I_,J_] = find(req{lvl});
    idx = ismemb(N*I(1:nz)+J(1:nz),sort(N*I_+J_));

    % update global sparse matrix
    M = sparse(I(idx),J(idx),S(idx),N,N);

    % print summary
    if opts.verb
      fprintf('%3d | %12d | %12d | %10.2e (s)\n',lvl,sum(idx),nnz(M),toc)
    end
  end

  % finish
  D = spdiags(M,0);
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end

  % sparse matrix access
  function A = spget_sk
    [A,P] = spget(M,sk,sk,P);
    if nsk && ~strcmpi(F.symm,'n')
      D_ = diag(diag(A));
      L_ = tril(A,-1);
      U_ = triu(A, 1);
      if strcmpi(F.symm,'s')
        A = D_ + L_ + L_.' + U_ + U_.';
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        A = D_ + L_ + L_' + U_ + U_';
      end
    end
  end
end