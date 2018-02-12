% ISOCTAVE   Check if running Octave instead of MATLAB.
%
%    R = ISOCTAVE() returns an indicator R for whether the environment is
%    Octave instead of MATLAB.

function r = isoctave()
  persistent x
  if isempty(x)
    x = exist('OCTAVE_VERSION','builtin');
  end
  r = x;
end