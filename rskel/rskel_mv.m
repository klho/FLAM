% RSKEL_MV  Multiply using recursive skeletonization.
%
%    Typical complexity for [M,N] = SIZE(A) with M >= N without loss of
%    generality: O(M + N) in 1D and O(M + N^(2*(1 - 1/D))) in D dimensions.
%
%    Y = RSKEL_MV(F,X) produces the matrix Y by applying the compressed matrix F
%    to the matrix X.
%
%    Y = RSKEL_MV(F,X,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also RSKEL.

function Y = rskel_mv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(rskel_mv(F,conj(X),'c')); return; end

  % initialize
  nlvl = F.nlvl;
  if F.symm == 'n' && trans == 'c', p = F.Q;
  else,                             p = F.P;
  end
  if F.symm == 'n' && trans == 'n', q = F.Q;
  else,                             q = F.P;
  end
  np = length(p); prem = true(np,1);  % which "left"  indices remain?
  nq = length(q); qrem = true(nq,1);  % which "right" indices remain?
  P = zeros(nq,2);     % permutation for before/after current level
  Z = cell(nlvl  ,1);  % successively compressed from   upward pass
  Y = cell(nlvl+1,1);  % successively    refined from downward pass

  % upward sweep
  P(q,1) = 1:nq;  % initial permutation
  pf = 0;         % index for current level (to avoid data copy)
  Z{1} = X(q,:);  % copy-permute from input
  for lvl = 1:nlvl-1

    % update permutation and copy-permute from lower level
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      if F.symm == 'n' && trans == 'n', qrem(F.U(i).crd) = false;
      else,                             qrem(F.U(i).rrd) = false;
      end
    end
    p1 =  pf + 1;
    p2 = ~pf + 1;
    pf = ~pf;
    P(q(qrem(q)),p2) = 1:nnz(qrem);
    Z{lvl+1}(P(qrem,p2),:) = Z{lvl}(P(qrem,p1),:);

    % apply interpolation operators
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      f = F.U(i);
      if F.symm == 'n' && trans == 'n'
        rd = P(f.crd,p1);
        sk = P(f.csk,p2);
      else
        rd = P(f.rrd,p1);
        sk = P(f.rsk,p2);
      end
      if F.symm == 'n'
        if trans == 'n', T = f.cT;
        else,            T = f.rT;
        end
      elseif F.symm == 's'
        if trans == 'n', T = conj(f.rT);
        else,            T =      f.rT ;
        end
      elseif F.symm == 'h', T = f.rT;
      end
      Z{lvl+1}(sk,:) = Z{lvl+1}(sk,:) + T*Z{lvl}(rd,:);
    end
  end

  % downward sweep
  prem(:) = false;
  P = zeros(np,2);  % reset permutations
  Q = zeros(nq,1);  % permutation for data from upward sweep
  pf = 0;
  nx = size(X,2);
  Y{nlvl+1} = zeros(0,nx);
  for lvl = nlvl:-1:1

    % update permutation and copy-permute from higher level
    r = p(prem(p));
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      if F.symm == 'n' && trans == 'c', prem(F.U(i).crd) = true;
      else,                             prem(F.U(i).rrd) = true;
      end
      if F.symm == 'n' && trans == 'n', qrem(F.U(i).crd) = true;
      else,                             qrem(F.U(i).rrd) = true;
      end
    end
    p1 =  pf + 1;
    p2 = ~pf + 1;
    pf = ~pf;
    np = nnz(prem);
    P(p(prem(p)),p1) = 1:np;
    Q(q(qrem(q))) = 1:nnz(qrem);
    Y{lvl} = zeros(np,nx);
    Y{lvl}(P(r,p1),:) = Y{lvl+1}(P(r,p2),:);

    % apply interpolation operators
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      f = F.U(i);
      if F.symm == 'n' && trans == 'c'
        rd  = P(f.crd,p1);
        sk1 = P(f.csk,p1);
        sk2 = P(f.csk,p2);
      else
        rd  = P(f.rrd,p1);
        sk1 = P(f.rsk,p1);
        sk2 = P(f.rsk,p2);
      end
      if F.symm == 'n'
        if trans == 'n', T = f.rT;
        else,            T = f.cT;
        end
      elseif F.symm == 's'
        if trans == 'n', T =      f.rT ;
        else,            T = conj(f.rT);
        end
      elseif F.symm == 'h', T = f.rT;
      end
      Y{lvl}(rd,:) = T'*Y{lvl+1}(sk2,:);
      Y{lvl}(sk1,:) = Y{lvl+1}(sk2,:);
    end

    % apply diagonal blocks
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      f = F.D(i);
      if trans == 'n'
        j = P(f.i,p1);
        k = Q(f.j);
        D = f.D;
      else
        j = P(f.j,p1);
        k = Q(f.i);
        D = f.D';
      end
      Y{lvl}(j,:) = Y{lvl}(j,:) + D*Z{lvl}(k,:);
    end
  end

  % extract output
  Y = Y{1}(P(:,p1),:);
end