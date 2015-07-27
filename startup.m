% add core paths
curpath = pwd;
dirs = {'core','geom','hifde','hifie','ifmm','mf','misc','quad','rskel', ...
        'rskelf'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% add auxiliary paths
dirs = {'hifde','mf','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s/mv',curpath,s{:}))
  addpath(sprintf('%s/%s/sv',curpath,s{:}))
end
dirs = {'hifie','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s/spdiag',curpath,s{:}))
end

% clear
clear