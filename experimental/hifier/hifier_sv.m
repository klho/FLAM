% HIFIER_SV  Solve using rectangular hierarchical interpolative factorization
%            for integral operators.
%
%    Y = HIFIER_SV(F,X) produces the matrix Y by applying the factored
%    pseudoinverse of the factored matrix F to the matrix X.
%
%    Y = HIFIE_SV(F,X,TRANS) computes Y = F\X if TRANS = 'N' (default),
%    Y = F.'\X if TRANS = 'T', and Y = F'\X if TRANS = 'C'.
%
%    See also HIFIE2R, HIFIER_MV.

function Y = hifier_sv(F,X,trans)

  % set default parameters
  if nargin < 3 || isempty(trans)
    trans = 'n';
  end

  % check inputs
  assert(strcmpi(trans,'n') || strcmpi(trans,'t') || strcmpi(trans,'c'), ...
         'FLAM:hifier_sv:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')

  % apply matrix factored pseudoinverse
  Y = rskelfr_sv(F,X,trans);
end