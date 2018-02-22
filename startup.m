% add core paths
curpath = pwd;
dirs = {'compat','core','geom','hifde','hifie','ifmm','mf','misc','quad', ...
        'rskel','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% add auxiliary paths
dirs = {'hifde','mf','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s/mv',curpath,s{:}))
  addpath(sprintf('%s/%s/sv',curpath,s{:}))
end
dirs = {'hifde','hifie','mf','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s/spdiag',curpath,s{:}))
end

% add experimental paths
dirs = {'experimental/rskelfr'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% clear
clear