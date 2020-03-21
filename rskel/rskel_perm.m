% RSKEL_PERM  Natural permutation for recursive skeletonization.
%
%    A "natural permutation" is a permutation such that all indices in a given
%    tree node and its descendants are contiguous. It can be obtained by a
%    preorder traversal of the tree.
%
%    [P,Q] = RSKEL_PERM(T) returns natural permutations P and Q for the row and
%    column indices, respectively, as denoted by the fields RXI and CXI in the
%    tree T.
%
%    [P,Q] = RSKEL_PERM(T,SYMM) only returns Q if SYMM = 'N' (default);
%    otherwise, Q = [].
%
%    See also HYPOCT, RSKEL.

function [P,Q] = rskel_perm(t,symm)

  % set default parameters
  if nargin < 2 || isempty(symm), symm = 'n'; end

  % initialize
  P = zeros(length([t.nodes.rxi]),1);
  if symm == 'n', Q = zeros(length([t.nodes.cxi]),1);
  else,           Q = [];
  end
  m = 0;  n = 0;  % running counters
  stack = zeros(t.lvp(end),1);
  stack(1) = 1;   % push root onto stack
  p = 1;          % current stack pointer

  % main loop
  while p > 0
    i = stack(p); p = p - 1;  % pop from stack

    % fill row permutation
    ri = t.nodes(i).rxi;
    rn = length(ri);
    P(m+(1:rn)) = ri;
    m = m + rn;

    % fill column permutation
    if symm == 'n'
      ci = t.nodes(i).cxi;
      cn = length(ci);
      Q(n+(1:cn)) = ci;
      n = n + cn;
    end

    % push children onto stack
    for j = t.nodes(i).chld
      p = p + 1;
      stack(p) = j;
    end
  end
end