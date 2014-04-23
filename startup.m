% add paths
curpath = pwd;
dirs = {'core','geom','hifde','hifie','ifmm','mf','misc','quad','rskel', ...
        'rskelf'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% clear
clear