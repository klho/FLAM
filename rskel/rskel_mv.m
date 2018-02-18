% RSKEL_MV   Multiply using recursive skeletonization.
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
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:rskel_mv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(rskel_mv(F,conj(X),'c'));
    return
  end

  % initialize
  M = F.M;
  N = F.N;
  nlvl = F.nlvl;
  rrem = true(M,1);
  crem = true(N,1);
  Z = cell(nlvl+1,1);
  Y = cell(nlvl+1,1);

  % upward sweep
  Z{1} = X;
  for lvl = 1:nlvl
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
    if strcmpi(trans,'n')
      Z{lvl+1} = Z{lvl}(pcrem1(find(crem)),:);
    else
      Z{lvl+1} = Z{lvl}(prrem1(find(rrem)),:);
    end

    % apply interpolation operators
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      if strcmpi(F.symm,'n')
        if strcmpi(trans,'n')
          rd = pcrem1(F.U(i).crd);
          sk = pcrem2(F.U(i).csk);
          T = F.U(i).cT;
        else
          rd = prrem1(F.U(i).rrd);
          sk = prrem2(F.U(i).rsk);
          T = F.U(i).rT';
        end
      elseif strcmpi(F.symm,'s')
        rd = pcrem1(F.U(i).rrd);
        sk = pcrem2(F.U(i).rsk);
        if strcmpi(trans,'n')
          T = F.U(i).rT.';
        else
          T = F.U(i).rT';
        end
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        rd = pcrem1(F.U(i).rrd);
        sk = pcrem2(F.U(i).rsk);
        T = F.U(i).rT';
      end
      Z{lvl+1}(sk,:) = Z{lvl+1}(sk,:) + T*Z{lvl}(rd,:);
    end
  end

  % downward sweep
  if strcmpi(trans,'n')
    Y{nlvl+1} = zeros(sum(rrem),size(X,2));
  else
    Y{nlvl+1} = zeros(sum(crem),size(X,2));
  end
  for lvl = nlvl:-1:1
    prrem2 = cumsum(rrem);
    pcrem2 = cumsum(crem);
    if strcmpi(trans,'n')
      rem_ = rrem;
    else
      rem_ = crem;
    end
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      rrem(F.U(i).rrd) = 1;
      if strcmpi(F.symm,'n')
        crem(F.U(i).crd) = 1;
      else
        crem(F.U(i).rrd) = 1;
      end
    end
    prrem1 = cumsum(rrem);
    pcrem1 = cumsum(crem);
    if strcmpi(trans,'n')
      Y{lvl} = zeros(sum(rrem),size(X,2));
      Y{lvl}(prrem1(find(rem_)),:) = Y{lvl+1};
    else
      Y{lvl} = zeros(sum(crem),size(X,2));
      Y{lvl}(pcrem1(find(rem_)),:) = Y{lvl+1};
    end

    % apply interpolation operators
    for i = F.lvpu(lvl)+1:F.lvpu(lvl+1)
      if strcmpi(F.symm,'n')
        if strcmpi(trans,'n')
          rd  = prrem1(F.U(i).rrd);
          sk1 = prrem1(F.U(i).rsk);
          sk2 = prrem2(F.U(i).rsk);
          T = F.U(i).rT;
        else
          rd  = pcrem1(F.U(i).crd);
          sk1 = pcrem1(F.U(i).csk);
          sk2 = pcrem2(F.U(i).csk);
          T = F.U(i).cT';
        end
      elseif strcmpi(F.symm,'s')
        rd  = prrem1(F.U(i).rrd);
        sk1 = prrem1(F.U(i).rsk);
        sk2 = prrem2(F.U(i).rsk);
        if strcmpi(trans,'n')
          T = F.U(i).rT;
        else
          T = conj(F.U(i).rT);
        end
      elseif strcmpi(F.symm,'h') || strcmpi(F.symm,'p')
        rd  = prrem1(F.U(i).rrd);
        sk1 = prrem1(F.U(i).rsk);
        sk2 = prrem2(F.U(i).rsk);
        T = F.U(i).rT;
      end
      Y{lvl}(rd,:) = T*Y{lvl+1}(sk2,:);
      Y{lvl}(sk1,:) = Y{lvl+1}(sk2,:);
    end

    % apply diagonal blocks
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      if strcmpi(trans,'n')
        j = prrem1(F.D(i).i);
        k = pcrem1(F.D(i).j);
        D = F.D(i).D;
      else
        j = pcrem1(F.D(i).j);
        k = prrem1(F.D(i).i);
        D = F.D(i).D';
      end
      Y{lvl}(j,:) = Y{lvl}(j,:) + D*Z{lvl}(k,:);
    end
  end

  % extract output
  Y = Y{1};
end