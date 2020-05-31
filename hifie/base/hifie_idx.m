% HIFIE_IDX  Compression function for HIFIE2X and HIFIE3X.

function [sk,rd,T] = hifie_idx(K,K1,K2,rank_or_tol,Tmax,rrqr_iter)
  n = size(K,2);

  % scale compression tolerance
  ratio = 1;
  if rank_or_tol < 1 && nnz(K2)
    nrm1 = snorm(n,@(x)(K1*x),@(x)(K1'*x));
    nrm2 = snorm(n,@(x)(K2*x),@(x)(K2'*x));
    ratio = min(1,nrm1/nrm2);
  end

  % partition by sparsity structure of modified interactions
  K2 = K2 ~= 0;
  K2 = K2(logical(sum(K2,2)),:);
  s = sum(K2);
  if sum(s) == 0
    grp = {1:n};
    ngrp = 1;
  else
    C = K2'*K2;         % Gramian for detecting nonzero overlap
    s = max(s',s);      % maximum nonzeros in either Gramian vector
    proc = false(n,1);  % already processed?
    grp = cell(n,1);
    ngrp = 0;
    for k = 1:n
      if proc(k), continue; end              % find remaining columns with ...
      idx = find(C(:,k) == s(:,k) & ~proc);  % ... fully matching sparsity ...
      if isempty(idx), continue; end         % ... pattern
      ngrp = ngrp + 1;
      grp{ngrp} = idx;
      proc(idx) = true;
    end
  end
  grp = grp(1:ngrp);

  % skeletonize by partition
  sk_ = cell(ngrp,1);
  rd_ = cell(ngrp,1);
  T_  = cell(ngrp,1);
  nsk = zeros(ngrp,1);
  nrd = zeros(ngrp,1);
  for k = 1:ngrp
    K_ = K(:,grp{k});
    [sk_{k},rd_{k},T_{k}] = id(K_,ratio*rank_or_tol,Tmax,rrqr_iter);
    nsk(k) = length(sk_{k});
    nrd(k) = length(rd_{k});
  end

  % reassemble skeletonization as block diagonal over partition
  psk = [0; cumsum(nsk(:))];  % index arrays
  prd = [0; cumsum(nrd(:))];
  sk = zeros(1,psk(end));
  rd = zeros(1,prd(end));
  T  = zeros(psk(end),prd(end));
  for k = 1:ngrp
    sk(psk(k)+1:psk(k+1)) = grp{k}(sk_{k});
    rd(prd(k)+1:prd(k+1)) = grp{k}(rd_{k});
     T(psk(k)+1:psk(k+1),prd(k)+1:prd(k+1)) = T_{k};
  end
end