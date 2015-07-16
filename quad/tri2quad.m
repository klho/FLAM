% TRI2QUAD   Double integral over a triangle in 2D in reference configuration.
%
%    Q = TRI2QUAD(FUN,V2,V3) approximates the integral of FUN over the triangle
%    with vertices [0; 0], [V2; 0], and V3.
%
%    Q = TRI2QUAD(FUN,V2,V3,PARAM1,VAL1,PARAM2,VAL2,...) passes optional
%    parameters to the underlying integration routine QUAD2D.
%
%    See also QUAD2D, TRI2TRANSROT, TRI3TRANSROT.

function q = tri2quad(fun,v2,v3,varargin)

  % integrate over left triangle on (0, V3(1))
  if abs(v3(1)) > 0
    a = 0;
    b = v3(1);
    c = 0;
    d = @(x)(v3(2)/v3(1)*x);
    q1 = quad2d(fun,a,b,c,d,varargin{:});
  else
    q1 = 0;
  end

  % integrate over right triangle on (V3(1), V2)
  if abs(v2 - v3(1)) > 0
    a = v3(1);
    b = v2;
    c = 0;
    d = @(x)(v3(2)/(v2 - v3(1))*(v2 - x));
    q2 = quad2d(fun,a,b,c,d,varargin{:});
  else
    q2 = 0;
  end

  % add integrals
  q = q1 + q2;
end