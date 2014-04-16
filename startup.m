% add paths
curpath = pwd;
dirs = {'core','geom','hifie','ifmm','misc','quad','rskel','rskelf'};
for s = dirs
  addpath(sprintf('%s/%s',curpath,s{:}))
end

% clear
clear