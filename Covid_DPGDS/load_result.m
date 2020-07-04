%%  load data and calculate the predicted error

file_name = '.\ \';
name_save = 'Covid__tria50MARE_Layer3_S0.mat';
load([file_name,name_save])

predict_mean= mean(estimate_error_meandimension,1);
predict_std = std(estimate_error_meandimension,1);
%% plot reconstruction samples
State = {'HI','AZ','IA','MN','OH','WV'};
INDEX = [13,   4,   17,  25,  37,  50]-1;

for fig_index =1: length(INDEX)
figure(fig_index);plot(X_train(INDEX(fig_index),2:end),'r','linewidth',2);hold on;
plot(poissrnd(Lambda(INDEX(fig_index),2:end)),'g','linewidth',2);hold on;
legend('ground truth','reconstruction');
set(gca,'FontSize',24);
title(State{fig_index});
end