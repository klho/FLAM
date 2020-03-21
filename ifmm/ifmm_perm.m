% IFMM_PERM  Natural permutation for interpolative fast multipole method.
%
%    A "natural permutation" is a permutation such that all indices in a given
%    tree node and its descendants are contiguous. It can be obtained by a
%    preorder traversal of the tree.
%
%    [P,Q] = IFMM_PERM(T) returns natural permutations P and Q for the row and
%    column indices, respectively, as denoted by the fields RXI and CXI in the
%    tree T.
%
%    [P,Q] = IFMM_PERM(T,SYMM) only returns Q if SYMM = 'N' (default);
%    otherwise, Q = [].
%
%    See also HYPOCT, IFMM.

function [P,Q] = ifmm_perm(t,symm)
  [P,Q] = rskel_perm(t,symm);
end