function [ config ] = initSynthetic(  )

%% data processing
load('/home/rahikalantari/MATLAB/Run_HBGP/data/synthetic10.mat');
%data = ans;
T = floor(.6*size(data,2));
TP = T;
config.rawMat = data(:,1:T,:);
config.dataMeanVec = mean(mean(data(:,1:T,:),3),2);
config.dataMat = data - repmat(config.dataMeanVec, 1, size(data, 2),size(data, 3));
config.pred_data = config.dataMat(:,T+1:end,:) ;
config.dataMat = config.dataMat(:,1:T,:)
dataMat = config.dataMat(:,1:T,:);
%pred_data = data_pred;
config.T = size(dataMat, 2);
config.numTs = size(dataMat, 1);


%% initalization config
config.hankelSizeOption = 1;
config.initLdsOption = 1;

%% EM configuration
config.maxIter = 600;

config.trainIdx = 1:config.T;
config.hiddenStates = [5,10,20];

config.sdgMaxIter = 600;
config.barrier = 1e-4;

% %% Sparse Group Lasso Lds config
% config.lambdaAs = 10.^[3:6];
% config.lambda2As = 10.^[3];
% config.lambda3As = [linspace(6000, 10000, 50) linspace(10^4, 10^6, 50)];
% config.valiRatio = 0.8;
% 
% 
% config.sdgThreshold = 1e-4;

end

