% IFMM_MV       Multiply using interpolative fast multipole method.
%
%    Y = IFMM_MV(F,X) produces the matrix Y by applying the compressed matrix F
%    to the matrix X. This assumes that all required interactions are stored in
%    F.
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
  if nargin < 3
    A = [];
  end
  if nargin < 4 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  if ~(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'))
    error('FLAM:ifmm_mv:invalidTrans', ...
          'Transpose parameter must be one of ''N'', ''T'', or ''C''.')
  end

  % handle transpose by conjugation
  if strcmpi(trans,'t')
    Y = conj(ifmm_mv(F,conj(X),A,'c'));
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
      Z{lvl+1} = Z{lvl}(pcrem1(crem),:);
    else
      Z{lvl+1} = Z{lvl}(prrem1(rrem),:);
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
      elseif strcmpi(F.symm,'h')
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
      Y{lvl}(prrem1(rem_),:) = Y{lvl+1};
    else
      Y{lvl} = zeros(sum(crem),size(X,2));
      Y{lvl}(pcrem1(rem_),:) = Y{lvl+1};
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
      elseif strcmpi(F.symm,'h')
        rd  = prrem1(F.U(i).rrd);
        sk1 = prrem1(F.U(i).rsk);
        sk2 = prrem2(F.U(i).rsk);
        T = F.U(i).rT;
      end
      Y{lvl}(rd,:) = T*Y{lvl+1}(sk2,:);
      Y{lvl}(sk1,:) = Y{lvl+1}(sk2,:);
    end

    % apply interaction matrices
    for i = F.lvpd(lvl)+1:F.lvpd(lvl+1)
      is = F.D(i).is;
      ie = F.D(i).ie;
      if strcmpi(F.symm,'n')
        js = F.D(i).js;
        je = F.D(i).je;
      else
        js = is;
        je = ie;
      end

      % get self-interactions
      if lvl == 1
        if strcmpi(F.store,'s') || strcmpi(F.store,'d') || strcmpi(F.store,'a')
          Ds = F.D(i).Ds;
        else
          Ds = A(is,js);
        end
      end

      % get external interactions
      if strcmpi(F.store,'a') || (lvl == 1 && strcmpi(F.store,'d'))
        Do = F.D(i).Do;
      else
        Do = A(ie,js);
      end
      if strcmpi(F.symm,'n')
        if strcmpi(F.store,'a') || (lvl == 1 && strcmpi(F.store,'d'))
          Di = F.D(i).Di;
        else
          Di = A(is,je);
        end
      elseif strcmpi(F.symm,'s')
        Di = Do.';
      elseif strcmpi(F.symm,'h')
        Di = Do';
      end

      % apply matrices
      is_ = is;
      js_ = js;
      ie_ = ie;
      je_ = je;
      if strcmpi(trans,'n')
        is = prrem1(is_);
        js = pcrem1(js_);
        io = prrem1(ie_);
        jo = pcrem1(js_);
        ii = prrem1(is_);
        ji = pcrem1(je_);
      else
        is = pcrem1(js_);
        js = prrem1(is_);
        io = pcrem1(js_);
        jo = prrem1(ie_);
        ii = pcrem1(je_);
        ji = prrem1(is_);
        if lvl == 1
          Ds = Ds';
        end
        Do = Do';
        Di = Di';
      end
      if lvl == 1
        Y{lvl}(is,:) = Y{lvl}(is,:) + Ds*Z{lvl}(js,:);
      end
      Y{lvl}(io,:) = Y{lvl}(io,:) + Do*Z{lvl}(jo,:);
      Y{lvl}(ii,:) = Y{lvl}(ii,:) + Di*Z{lvl}(ji,:);
    end
  end

  % extract output
  Y = Y{1};
end