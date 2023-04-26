% RSKEL_XSP  Extended sparsification for recursive skeletonization.
%
%    The extended sparsification of a compressed matrix F = D + U*S*V' is a
%    sparse matrix embedding
%
%          [D   U   ]
%      A = [V'    -I]
%          [   -I  S]
%
%    where I is an identity matrix of the appropriate size and S is itself
%    possibly expanded in the same way, following the hierarchical structure of
%    the compressed representation. This sparse form can be used to solve linear
%    systems and least squares problems, e.g., the solution of F*X = B can be
%    recovered from that of A*[X; Y; Z] = [B; 0; 0], where B is zero-padded as
%    required and A can be factored efficiently using standard sparse direct
%    solvers.
%
%    If F.SYMM = 'N', then the entire extended sparse matrix is returned;
%    otherwise, only the lower triangular part is returned.
%
%    Typical complexity: same as RSKEL_MV.
%
%    A = RSKEL_XSP(F) produces the extended sparsification A of the compressed
%    matrix F.
%
%    [A,P,Q] = RSKEL_XSP(F) returns the extended sparsification in "natural"
%    order according to the row and column permutations P and Q, respectively,
%    such that each submatrix D, U, V, S, etc. is block diagonal. This ordering
%    better exposes the inherent structure in A and can lead to more efficient
%    sparse factorizations. If [M,N] = SIZE(F) and B = RSKEL_XSP(F), then
%    A(1:M,1:N) = B(P,Q). Note that the permutations only pertain to the
%    original non-extended indices; the others are defined internally in a self-
%    consistent way.
%
%    See also RSKEL.

function [A,p,q] = rskel_xsp(F)

  % initialize
  nlvl = F.nlvl;
  P = zeros(F.M,2);  % row permutations for current/next level
  Q = zeros(F.N,2);  % col permutations for current/next level
  if nargout < 2, p = 1:F.M;    % default row ordering
  else,           p = F.P;      % natural row ordering
  end
  if nargout < 3, q = 1:F.N;    % default col ordering
  else
    if F.symm == 'n', q = F.Q;  % natural col ordering
    else,             q = F.P;
    end
  end
  P(p) = 1:F.M; Q(q) = 1:F.N;  % initial permutations
  pf = 0;                      % index for current level (to avoid data copy)
  M = 0;
  N = 0;

  % allocate storage
  rrem = true(F.M,1);
  crem = true(F.N,1);
  nz = 0;  % total number of nonzeros
  for lvl = 1:nlvl
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1), nz = nz + numel(F.D(i).D); end
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      f = F.U(i);
      rrem(f.rrd) = false;
      if F.symm == 'n'
        crem(f.crd) = false;
        nz = nz + numel(f.rT) + numel(f.cT);
      else
        crem(f.rrd) = false;
        nz = nz + numel(f.rT);
      end
    end
    if F.symm == 'n', nz = nz + 2*(nnz(rrem) + nnz(crem));
    else,             nz = nz +    nnz(rrem) + nnz(crem);
    end
  end
  I = zeros(nz,1);
  J = zeros(nz,1);
  S = zeros(nz,1);
  nz = 0;
  rrem(:) = true;
  crem(:) = true;

  % loop over levels
  for lvl = 1:nlvl

    % compute index data
    rn = nnz(rrem);
    cn = nnz(crem);
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      f = F.U(i);
      rrem(f.rrd) = false;
      if F.symm == 'n', crem(f.crd) = false;
      else,             crem(f.rrd) = false;
      end
    end
    rk = nnz(rrem);
    ck = nnz(crem);
    p1 =  pf + 1;
    p2 = ~pf + 1;
    pf = ~pf;
    P(p(rrem(p)),p2) = 1:rk;
    Q(q(crem(q)),p2) = 1:ck;

    % embed diagonal matrices
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      f = F.D(i);
      [j,k] = ndgrid(f.i,f.j);
      D = f.D;
      n = numel(D);
      I(nz+(1:n)) = M + P(j(:),p1);
      J(nz+(1:n)) = N + Q(k(:),p1);
      S(nz+(1:n)) = D(:);
      nz = nz + n;
    end

    % terminate if at root
    if lvl == nlvl
      M = M + rn;
      N = N + cn;
      break
    end

    % embed interpolation identity matrices
    if F.symm == 'n'
      I(nz+(1:rk)) = M + P(find(rrem),p1);
      J(nz+(1:rk)) = N + cn + P(find(rrem),p2);
      S(nz+(1:rk)) = ones(rk,1);
      nz = nz + rk;
    end
    I(nz+(1:ck)) = M + rn + Q(find(crem),p2);
    J(nz+(1:ck)) = N + Q(find(crem),p1);
    S(nz+(1:ck)) = ones(ck,1);
    nz = nz + ck;

    % embed interpolation matrices
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      f = F.U(i);
      rrd = f.rrd;
      rsk = f.rsk;
      rT  = f.rT';
      if F.symm == 'n'
        crd = f.crd;
        csk = f.csk;
        cT  = f.cT;
      elseif F.symm == 's'
        crd = f.rrd;
        csk = f.rsk;
        cT  = rT.';
      elseif F.symm == 'h'
        crd = f.rrd;
        csk = f.rsk;
        cT  = rT';
      end

      % row interpolation
      if F.symm == 'n'
        [j,k] = ndgrid(rrd,rsk);
        n = numel(rT);
        I(nz+(1:n)) = M + P(j(:),p1);
        J(nz+(1:n)) = N + cn + P(k(:),p2);
        S(nz+(1:n)) = rT(:);
        nz = nz + n;
      end

      % column interpolation
      [j,k] = ndgrid(csk,crd);
      n = numel(cT);
      I(nz+(1:n)) = M + rn + Q(j(:),p2);
      J(nz+(1:n)) = N + Q(k(:),p1);
      S(nz+(1:n)) = cT(:);
      nz = nz + n;
    end

    % embed identity matrices
    M = M + rn;
    N = N + cn;
    if F.symm == 'n'
      I(nz+(1:ck)) = M + (1:ck);
      J(nz+(1:ck)) = N + rk + (1:ck);
      S(nz+(1:ck)) = -ones(ck,1);
      nz = nz + ck;
    end
    I(nz+(1:rk)) = M + ck + (1:rk);
    J(nz+(1:rk)) = N + (1:rk);
    S(nz+(1:rk)) = -ones(rk,1);
    nz = nz + rk;

    % move pointer to next level
    M = M + ck;
    N = N + rk;
  end

  % assemble sparse matrix
  if F.symm ~= 'n'
    idx = I >= J;
    I = I(idx);
    J = J(idx);
    S = S(idx);
  end
  A = sparse(I,J,S,M,N);
end