% HIFIE_DIAG   Extract diagonal using hierarchical interpolative factorization
%              for integral equations via matrix unfolding.
%
%    D = HIFIE_DIAG(F) produces the diagonal D of the factored matrix F.
%
%    D = HIFIE_DIAG(F,DINV) computes D = DIAG(F) if DINV = 0 (default) and
%    D = DIAG(INV(F)) if DINV = 1.
%
%    D = HIFIE_DIAG(F,DINV,OPTS) also passes various options to the algorithm.
%    Valid options include:
%
%      - VERB: display status of the code if VERB = 1 (default: VERB = 0).
%
%    See also HIFIE2, HIFIE2X, HIFIE3, HIFIE3X, HIFIE_SPDIAG.

function D = hifie_diag(F,dinv,opts)
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

    % store tranpose requirement matrix from previous level for quick row access
    req_ = req{lvl}';

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

      % require external entries
      if lvl > 1
        rd = F.factors(i).rd;
        [I_,~] = find(req{lvl}(:,sk));
        [J_,~] = find(req_(:,sk));
        I_ = union(I_,J_);
        ex = I_(~ismemb(I_,sort([rd sk])));
        ex = ex(rem(ex));
        [I_,J_] = ndgrid(ex,sk);
        I__ = I_(:);
        J__ = J_(:);
        I_ = [I__; J__];
        J_ = [J__; I__];
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
    end

    % construct requirement matrix
    idx = 1:nz;
    if strcmpi(F.symm,'s') || strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      idx = find(I(idx) >= J(idx));
    end
    req{lvl+1} = logical(sparse(I(idx),J(idx),ones(size(idx)),N,N));
  end

  % allocate storage for all required entries (needed for on-the-fly updating)
  req_ = logical(sparse(N,N));
  for lvl = 1:nlvl
    req_ = req_ | req{lvl};
  end
  [I,J] = find(req_);
  Ap = [0; cumsum(histc(J,1:N))];
  e = cell(N,1);
  Ai = e;
  Ax = e;
  for i = 1:N
    Ai{i} = I(Ap(i)+1:Ap(i+1));
    Ax{i} = zeros(Ap(i+1)-Ap(i),1);
  end

  % print summary
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    fprintf('%3s | %12d | %25.2e (s)\n','-',nnz(req_),toc)
  end

  % unfold factorization
  for lvl = nlvl:-1:1
    tic
    req_ = req{lvl}';
    [I_,J_] = find(req{lvl});
    remidx = sort(I_+N*J_);
    nzget = 0;
    nzadd = 0;

    % loop over nodes
    for i = F.lvp(lvl)+1:F.lvp(lvl+1)
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      rem(rd) = 1;
      T = F.factors(i).T;
      L = F.factors(i).L;
      E = F.factors(i).E;
      if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
        G = F.factors(i).F;
        U = F.factors(i).U;
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        G = F.factors(i).E';
        U = F.factors(i).L';
      end

      % find external interactions
      [I_,~] = find(req{lvl}(:,sk));
      [J_,~] = find(req_(:,sk));
      I_ = union(I_,J_);
      ex = I_(~ismemb(I_,sort([rd sk])))';
      ex = ex(rem(ex));

      % compute local matrix
      nrd = length(rd);
      nsk = length(sk);
      nex = length(ex);
      ird = 1:nrd;
      isk = nrd+1:nrd+nsk;
      iex = nrd+nsk+1:nrd+nsk+nex;
      ssk = sort(sk);
      sex = sort(ex);
      D = zeros(nrd+nsk+nex);
      D(isk,isk) = spget_sk;
      [D(iex,isk),D(isk,iex)] = spget_ex;
      D_ = D;
      if strcmpi(F.symm,'h')
        if dinv
          D(ird,ird) = inv(F.factors(i).U);
        else
          D(ird,ird) = F.factors(i).U;
        end
      else
        D(ird,ird) = eye(nrd);
      end
      if dinv
        D(:,ird) = (D(:,ird) - D(:,isk)*E)/L;
        D(ird,:) = U\(D(ird,:) - G*D(isk,:));
      else
        D(:,isk) = D(:,isk) + D(:,ird)*G;
        D(isk,:) = D(isk,:) + E*D(ird,:);
      end
      if dinv
        if strcmp(F.symm,'s')
          D(:,isk) = D(:,isk) - D(:,ird)*T.';
        else
          D(:,isk) = D(:,isk) - D(:,ird)*T';
        end
        D(isk,:) = D(isk,:) - T*D(ird,:);
      else
        D(:,ird) = D(:,ird)*U + D(:,isk)*T;
        if strcmp(F.symm,'s')
          D(ird,:) = L*D(ird,:) + T.'*D(isk,:);
        else
          D(ird,:) = L*D(ird,:) + T'*D(isk,:);
        end
      end
      D = D - D_;

      % keep only remaining entries
      [I,J] = ndgrid([rd sk ex]);
      idx = ismemb(I+N*J,remidx);
      I = I(idx);
      J = J(idx);
      D = D(idx);

      % update global sparse matrix
      spadd(I,J,D);
    end

    % print summary
    if opts.verb
      fprintf('%3d | %12d | %12d | %10.2e (s)\n',lvl,nzget,nzadd,toc)
    end
  end

  % finish
  D = zeros(N,1);
  for i = 1:N
    D(i) = Ax{i}(Ai{i} == i);
  end
  if opts.verb
    fprintf([repmat('-',1,80) '\n'])
    toc(start)
  end

  % sparse matrix add
  function spadd(I,J,S)
    K = unique(J);
    Bp = [0; cumsum(histc(J,K))];
    n = length(K);
    e = cell(n,1);
    Bi = e;
    Bx = e;
    [~,idx] = sort(I + N*J);
    I = I(idx);
    S = S(idx);
    for i = 1:n
      Bi{i} = I(Bp(i)+1:Bp(i+1));
      Bx{i} = S(Bp(i)+1:Bp(i+1));
    end
    for j = 1:n
      idx = ismemb(Ai{K(j)},Bi{j});
      Ax{K(j)}(idx) = Ax{K(j)}(idx) + Bx{j};
      nzadd = nzadd + sum(idx);
    end
  end

  % get (external, skeleton) block
  function A = spget_es
    A = zeros(nex,nsk);
    P(ex) = 1:nex;
    for j = 1:nsk
      Ai_ = Ai{sk(j)};
      idx = ismemb(Ai_,sex);
      A(P(Ai_(idx)),j) = Ax{sk(j)}(idx);
      nzget = nzget + sum(idx);
    end
  end

  % get external block matrices
  function [Aes,Ase] = spget_ex
    Aes = spget_es;
    Ase = spget_se;
    if ~strcmpi(F.symm,'n')
      if strcmpi(F.symm,'s')
        Aes = Aes + Ase.';
        Ase = Aes.';
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        Aes = Aes + Ase';
        Ase = Aes';
      end
    end
  end

  % get (skeleton, external) block
  function A = spget_se
    A = zeros(nsk,nex);
    P(sk) = 1:nsk;
    for j = 1:nex
      Ai_ = Ai{ex(j)};
      idx = ismemb(Ai_,ssk);
      A(P(Ai_(idx)),j) = Ax{ex(j)}(idx);
      nzget = nzget + sum(idx);
    end
  end

  % get (skeleton, skeleton) block
  function A = spget_sk
    A = zeros(nsk);
    P(sk) = 1:nsk;
    for j = 1:nsk
      Ai_ = Ai{sk(j)};
      idx = ismemb(Ai_,ssk);
      A(P(Ai_(idx)),j) = Ax{sk(j)}(idx);
      nzget = nzget + sum(idx);
    end
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