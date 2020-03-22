% HYPOCT_PERM  Natural tree permutation.
%
%    A "natural permutation" is a permutation such that all points in any given
%    subtree are contiguous. It can be obtained by a preorder traversal of the
%    tree.
%
%    P = HYPOCT_PERM(T) returns a natural permutation P for the tree T.
%
%    See also HYPOCT.

function P = hypoct_perm(t)

  % initialize
  P = zeros(length([t.nodes.xi]),1);
  n = 0;          % running index
  stack = zeros(t.lvp(end),1);
  stack(1) = 1;   % push root onto stack
  p = 1;          % current stack pointer

  % main loop
  while p > 0

    % pop from stack
    i = stack(p);
    p = p - 1;

    % fill permutation
    xi = t.nodes(i).xi;
    m = length(xi);
    P(n+(1:m)) = xi;
    n = n + m;

    % push children onto stack
    for j = t.nodes(i).chld
      p = p + 1;
      stack(p) = j;
    end
  end
end