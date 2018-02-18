% RSKEL_XSP  Extended sparsification for recursive skeletonization.
%
%    A = RSKEL_XSP(F) produces the extended sparsification A of the compressed
%    matrix F. If F has the single-level representation D + U*S*V', then
%    A = [D U 0; V' 0 -I; 0 -I S], where I is an identity matrix of the
%    appropriate size; in the multilevel setting, S itself is extended in the
%    same way. If F.SYMM = 'N', then the entire extended sparsification is
%    returned; if F.SYMM = 'S', 'H', or 'P', then only the lower triangular part
%    part of A is returned.
%
%    See also RSKEL.

function A = rskel_xsp(F)

  % initialize
  nlvl = F.nlvl;
  M = 0;
  N = 0;

  % allocate storage
  rrem = true(F.M,1);
  crem = true(F.N,1);
  nz = 0;
  for lvl = 1:nlvl
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      nz = nz + numel(F.D(i).D);
    end
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      rrem(F.U(i).rrd) = 0;
      if strcmpi(F.symm,'n')
        crem(F.U(i).crd) = 0;
        nz = nz + numel(F.U(i).rT) + numel(F.U(i).cT);
      elseif strcmpi(F.symm,'s')
        crem(F.U(i).rrd) = 0;
        nz = nz + 2*numel(F.U(i).rT);
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        crem(F.U(i).rrd) = 0;
        nz = nz + numel(F.U(i).rT);
      end
    end
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      nz = nz + 2*(sum(rrem) + sum(crem));
    elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
      nz = nz + sum(rrem) + sum(crem);
    end
  end
  I = zeros(nz,1);
  J = zeros(nz,1);
  S = zeros(nz,1);
  nz = 0;
  rrem(:) = 1;
  crem(:) = 1;

  % loop over levels
  for lvl = 1:nlvl

    % compute index data
    prrem1 = cumsum(rrem);
    pcrem1 = cumsum(crem);
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      rrem(F.U(i).rrd) = 0;
      if strcmpi(F.symm,'n')
        crem(F.U(i).crd) = 0;
      else
        crem(F.U(i).rrd) = 0;
      end
    end
    prrem2 = cumsum(rrem);
    pcrem2 = cumsum(crem);
    rn = prrem1(end);
    cn = pcrem1(end);
    rk = prrem2(end);
    ck = pcrem2(end);

    % embed diagonal matrices
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      [j,k] = ndgrid(F.D(i).i,F.D(i).j);
      D = F.D(i).D;
      m = numel(D);
      I(nz+1:nz+m) = M + prrem1(j(:));
      J(nz+1:nz+m) = N + pcrem1(k(:));
      S(nz+1:nz+m) = D(:);
      nz = nz + m;
    end

    % terminate if at root
    if lvl == nlvl
      M = M + rn;
      N = N + cn;
      break
    end

    % embed interpolation identity matrices
    I(nz+1:nz+rk) = M + prrem1(find(rrem));
    J(nz+1:nz+rk) = N + cn + prrem2(find(rrem));
    S(nz+1:nz+rk) = ones(rk,1);
    nz = nz + rk;
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      I(nz+1:nz+ck) = M + rn + pcrem2(find(crem));
      J(nz+1:nz+ck) = N + pcrem1(find(crem));
      S(nz+1:nz+ck) = ones(ck,1);
      nz = nz + ck;
    end

    % embed interpolation matrices
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      rrd = F.U(i).rrd;
      rsk = F.U(i).rsk;
      rT  = F.U(i).rT;
      if strcmpi(F.symm,'n')
        crd = F.U(i).crd;
        csk = F.U(i).csk;
        cT  = F.U(i).cT;
      elseif strcmpi(F.symm,'s')
        crd = F.U(i).rrd;
        csk = F.U(i).rsk;
        cT  = F.U(i).rT.';
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        crd = F.U(i).rrd;
        csk = F.U(i).rsk;
        cT  = F.U(i).rT';
      end

      % row interpolation
      [j,k] = ndgrid(rrd,rsk);
      m = numel(rT);
      I(nz+1:nz+m) = M + prrem1(j(:));
      J(nz+1:nz+m) = N + cn + prrem2(k(:));
      S(nz+1:nz+m) = rT(:);
      nz = nz + m;

      % column interpolation
      if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
        [j,k] = ndgrid(csk,crd);
        m = numel(cT);
        I(nz+1:nz+m) = M + rn + pcrem2(j(:));
        J(nz+1:nz+m) = N + pcrem1(k(:));
        S(nz+1:nz+m) = cT(:);
        nz = nz + m;
      end
    end

    % embed identity matrices
    M = M + rn;
    N = N + cn;
    if strcmpi(F.symm,'n') || strcmpi(F.symm,'s')
      I(nz+1:nz+rk) = M + ck + (1:rk);
      J(nz+1:nz+rk) = N + (1:rk);
      S(nz+1:nz+rk) = -ones(rk,1);
      nz = nz + rk;
    end
    I(nz+1:nz+ck) = M + (1:ck);
    J(nz+1:nz+ck) = N + rk + (1:ck);
    S(nz+1:nz+ck) = -ones(ck,1);
    nz = nz + ck;

    % move pointer to next level
    M = M + ck;
    N = N + rk;
  end

  % assemble sparse matrix
  if strcmpi(F.symm,'n')
    A = sparse(I,J,S,M,N);
  elseif strcmpi(F.symm,'s')
    idx = I <= J;
    I = I(idx);
    J = J(idx);
    S = S(idx);
    A = sparse(J,I,S,M,N);
  elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
    idx = I <= J;
    I = I(idx);
    J = J(idx);
    S = S(idx);
    A = sparse(J,I,conj(S),M,N);
  end
end