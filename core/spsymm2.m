% SPSYMM2  Sparse matrix symmetrization for off-diagonal blocks.
%
%   [A,B] = SPSYMM2(A,B,SYMM) returns:
%
%     - A := A       and  B := B   if SYMM = 'N'.
%     - A := A + B.' then B := A.' if SYMM = 'S'.
%     - A := A + B'  then B := A'  if SYMM = 'H' or 'P'.
%
%   In all cases, A and B are related symmetrically as specified by SYMM.
%
%   See also SPSYMM.

function [A,B] = spsymm2(A,B,symm)
  if symm == 'n', return; end
  if symm == 's', A = A + B.'; B = A.';
  else,           A = A + B' ; B = A' ;
  end
end