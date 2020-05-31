% IFMM_MV  Multiply using interpolative fast multipole method.
%
%    Typical complexity: same as IFMM.
%
%    Y = IFMM_MV(F,X) produces the matrix Y by applying the compressed matrix F
%    to the matrix X. This assumes that all required interactions are stored in
%    F (i.e., IFMM was used with option STORE = 'A').
%
%    Y = IFMM_MV(F,X,A) computes Y by generating all missing required
%    interactions from A.
%
%    Y = IFMM_MV(F,X,A,TRANS) computes Y = F*X if TRANS = 'N' (default),
%    Y = F.'*X if TRANS = 'T', and Y = F'*X if TRANS = 'C'.
%
%    See also IFMM.

function Y = ifmm_mv(F,X,A,trans)

  % set default parameters
  if nargin < 3, A = []; end
  if nargin < 4 || isempty(trans), trans = 'n'; end

  % check inputs
  trans = chktrans(trans);

  % handle transpose by conjugation
  if trans == 't', Y = conj(ifmm_mv(F,conj(X),A,'c')); return; end

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
      if F.symm == 'n' && trans == 'n'
        rd = P(F.U(i).crd,p1);
        sk = P(F.U(i).csk,p2);
      else
        rd = P(F.U(i).rrd,p1);
        sk = P(F.U(i).rsk,p2);
      end
      if F.symm == 'n'
        if trans == 'n', T = F.U(i).cT;
        else,            T = F.U(i).rT;
        end
      elseif F.symm == 's'
        if trans == 'n', T = conj(F.U(i).rT);
        else,            T =      F.U(i).rT ;
        end
      elseif F.symm == 'h', T = F.U(i).rT;
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
      if F.symm == 'n' && trans == 'c'
        rd  = P(F.U(i).crd,p1);
        sk1 = P(F.U(i).csk,p1);
        sk2 = P(F.U(i).csk,p2);
      else
        rd  = P(F.U(i).rrd,p1);
        sk1 = P(F.U(i).rsk,p1);
        sk2 = P(F.U(i).rsk,p2);
      end
      if F.symm == 'n'
        if trans == 'n', T = F.U(i).rT;
        else,            T = F.U(i).cT;
        end
      elseif F.symm == 's'
        if trans == 'n', T =      F.U(i).rT ;
        else,            T = conj(F.U(i).rT);
        end
      elseif F.symm == 'h', T = F.U(i).rT;
      end
      Y{lvl}(rd,:) = T'*Y{lvl+1}(sk2,:);
      Y{lvl}(sk1,:) = Y{lvl+1}(sk2,:);
    end

    % apply interaction matrices
    for i = F.lvpb(lvl)+1:F.lvpb(lvl+1)
      is = F.B(i).is;
      ie = F.B(i).ie;
      if F.symm == 'n'
        js = F.B(i).js;
        je = F.B(i).je;
      else
        js = is;
        je = ie;
      end

      % get self-interactions
      if lvl == 1
        if F.store == 'n', D = A(is,js);
        else,              D = F.B(i).D;
        end

      % get external interactions
      else
        near = lvl == 2 && F.store == 'r';  % near field stored?
        if F.store == 'a' || near, Bo = F.B(i).Bo;
        else,                      Bo = A(ie,js);
        end
        if F.symm == 'n'
          if F.store == 'a' || near, Bi = F.B(i).Bi;
          else,                      Bi = A(is,je);
          end
        elseif F.symm == 's', Bi = Bo.';
        elseif F.symm == 'h', Bi = Bo';
        end
      end

      % apply matrices
      is_ = is; js_ = js;
      ie_ = ie; je_ = je;
      if trans == 'n'
        is = P(is_,p1); js = Q(js_);
        io = P(ie_,p1); jo = Q(js_);
        ii = P(is_,p1); ji = Q(je_);
      else
        is = P(js_,p1); js = Q(is_);
        io = P(js_,p1); jo = Q(ie_);
        ii = P(je_,p1); ji = Q(is_);
        if lvl == 1
          D = D';
        else
          Bo = Bo';
          Bi = Bi';
        end
      end
      if lvl == 1
        Y{lvl}(is,:) = Y{lvl}(is,:) + D*Z{lvl}(js,:);
      else
        Y{lvl}(io,:) = Y{lvl}(io,:) + Bo*Z{lvl}(jo,:);
        Y{lvl}(ii,:) = Y{lvl}(ii,:) + Bi*Z{lvl}(ji,:);
      end
    end
  end

  % extract output
  Y = Y{1}(P(:,p1),:);
end