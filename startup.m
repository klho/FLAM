% add paths
curpath = pwd;
dirs = {'core','geom','hifie','ifmm','mf','misc','quad','rskel','rskelf', ...
        'util'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% clear
clear