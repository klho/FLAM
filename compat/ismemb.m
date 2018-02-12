% ISMEMB  Fast dispatch for ISMEMBER.
%
%    IDX = ISMEMB(A,S) returns the output IDX of ISMEMBC(A,S) if in MATLAB and
%    otherwise the equivalent but slower ISMEMBER(A,S).

function idx = ismemb(A,S)
  if isoctave()
    idx = ismember(A,S);
  else
    idx = ismembc(A,S);
  end
end