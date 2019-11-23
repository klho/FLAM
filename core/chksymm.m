% CHKSYMM  Check symmetry parameter.
%
%    SYMM = CHKSYMM(SYMM) validates the symmetry parameter SYMM and converts it
%    to lowercase.
%
%    See also CHKTRANS.

function symm = chksymm(symm)
  symm = lower(symm);
  assert(strcmp(symm,'n') || strcmp(symm,'s') || strcmp(symm,'h') || ...
         strcmp(symm,'p'),'FLAM:chksymm:invalidSymm', ...
         'Symmetry parameter must be one of ''N'', ''S'', ''H'', or ''P''.')
end