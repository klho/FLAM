% DETPERM  Determinant of permutation vector.
%
%    D = DETPERM(P) computes the determinant (i.e., sign) D of the permutation
%    P.
%
%    In both MATLAB and Octave, this is done by calling DET on the associated
%    permutation matrix, which for efficiency should be represented explicitly
%    as sparse in the former but as dense (internally recognized as a
%    permutation matrix) in the latter.
%
%    The determinant can be computed in linear time, which both the MATLAB and
%    Octave implementations seem to achieve.

function d = detperm(p)
  n = length(p);
  if isoctave(), P =   eye(n);  % Octave uses special representation
  else,          P = speye(n);
  end
  d = det(P(:,p));
end