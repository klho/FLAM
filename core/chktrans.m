% CHKTRANS  Check transpose parameter.
%
%    TRANS = CHKTRANS(TRANS) validates the transpose parameter TRANS and
%    converts it to lowercase.
%
%    See also CHKSYMM.

function trans = chktrans(trans)
  trans = lower(trans);
  assert(strcmp(trans,'n') || strcmp(trans,'t') || strcmp(trans,'c'), ...
         'FLAM:chktrans:invalidTrans', ...
         'Transpose parameter must be one of ''N'', ''T'', or ''C''.')
end