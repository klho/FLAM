% SPSYMM  Sparse matrix symmetrization.
%
%   A = SPSYMM(A,SYMM) returns:
%
%     - A := A                       if SYMM = 'N'.
%     - A := A + A.' - DIAG(DIAG(A)) if SYMM = 'S'.
%     - A := A + A'  - DIAG(DIAG(A)) if SYMM = 'H' or 'P'.
%
%   This is useful for recovering a full matrix from a compactly stored one in
%   which only roughly half of the nonzeros are stored because of symmetry.
%
%   See also SPSYMM2.

function A = spsymm(A,symm)
  if isempty(A) || symm == 'n', return; end
  D = diag(diag(A));
  if symm == 's', A = A + A.' - D;
  else,           A = A + A'  - D;
  end
end