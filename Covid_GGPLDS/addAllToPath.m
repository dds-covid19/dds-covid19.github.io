function [] = addAllToPath()
%ADDLDSTOPATH Add the file from LDS package to Matlab path

disp('adding lds package into matlab path...');

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located

addpath([mydir,'Kalman_All/KPMstats'])
addpath([mydir,'Kalman_All/KPMtools'])
addpath([mydir,'Kalman_All/Kalman'])
addpath([mydir,'Kalman_All'])

%addpath(genpath([mydir, 'grouplasso_1_0']))
addpath([mydir, 'lib']);

addpath(mydir);

clear me mydir

end

