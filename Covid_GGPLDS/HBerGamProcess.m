clc
clear all;
close all;
addAllToPath()
%chose the dataset
dataset_name = 'Pedestrian'% 'Lorenz','Pedestrian','Stock'
%chose the task
task = 'predeiction'% 'interpretation','predeiction'
% data reconstruction (time consuming due to Kalman smoothing)
data_reconstruction_flag = 'yes'% 'yes','no'



Folder_name = 'results';
if strcmp(dataset_name,'Lorenz')
    load('data/Lorenz1.mat')
    data5=data;
    if strcmp(task,'interpretation')
        File_name = 'Lorenz_model_K_30_K_prime16_interpretation'
        portion = 0.9225;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 8.;%1e-2%*ones(K,1);
    elseif strcmp(task,'predeiction')
        File_name = 'Lorenz_model_K_30_K_prime16_prediction'
        portion = 0.77;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 0.1;%1e-2%*ones(K,1);
    end
    
elseif strcmp(dataset_name,'Pedestrian')
    load('data/3DMOT2015/train/PETS09-S2L1/gt/groud_truth_part.mat')
    data5=data;
    if strcmp(task,'interpretation')
        File_name = 'Pedesrian_model_K_30_K_prime16_interpretation'
        portion = 0.9091;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 1.;%1e-2%*ones(K,1);
    elseif strcmp(task,'predeiction')
        File_name = 'Pedestrian_model_K_30_K_prime16_prediction'
        portion = 0.9091;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 1.;%1e-2%*ones(K,1);
    end
    
elseif strcmp(dataset_name,'Stock')
    load('data/stock12corps.mat')
    data5=data2;
    if strcmp(task,'interpretation')
        File_name = 'Stock_model_K_30_K_prime16_interpretation'
        portion = 0.8440;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 0.48;%1e-2%*ones(K,1);
    elseif strcmp(task,'predeiction')
        File_name = 'Stock_model_K_30_K_prime16_prediction'
        portion = 0.8440;
        Model_hyper_params.alpha1 = 1;
        Model_hyper_params.beta1 = 0.48;%1e-2%*ones(K,1);
    end
    
else
    print('Wrong set up parameters')
    quit
end

Model_properties.K = 30;
Model_properties.K_prime = 16;
K= Model_properties.K;
K_prime = Model_properties.K_prime;
run_number = 1;
rng(run_number);
%K_S = 4;

T_S = floor(portion*size(data5,2));
T_P = size(data5,2)-T_S;
for i = 1:size(data5,2)
    if i <= T_S
        X{i} = repmat(data5(:,i),1,1);
    else
        X_w{i-T_S} = repmat(data5(:,i),1,1);
    end
end
T = T_S;
Model_properties.T = T;

Model_properties.X_dim_S = size(data5,1);
X_dim_S = Model_properties.X_dim_S;
Synt_data_type ="mine";
initilization = "random";


aaa1 =0;
for i=1:length(X)
    N(i) = size(X{i},2);
    MM(:,i)= mean(X{i},2);
    mean_plot(:,i) = mean(X{i},2);
end
MMM = mean(MM,2);
for i=1:length(X)
    X{i}= X{i}-MMM;
end
for i=1:length(X_w)
    X_w{i}= X_w{i}-MMM;
end
for i = 1:size(mean_plot,1)
    subplot(ceil(size(mean_plot,1)/3),ceil(size(mean_plot,1)/ceil(size(mean_plot,1)/3)),i)
    plot(mean_plot(i,:))
    title(sprintf('observation dimension %d ',i));
    legend('Real')
end
Model_properties.N =N;
Model_properties.N_sum = sum(N);
figure();
Sparsity_ratio = (0.48)*rand(1)+0.08;
N_sum = sum(N);
N_prod = prod(N);
Max_MCMC = 6000;
Burnin = 3000;
%K = 4;
T = T_S;
% the Model starts here
% Folowing are model assumption : 
% X(t)~N(D*s(t),Phi^-1), D(k)~N(0,Lambda_k^-1), Lambda_k^-1 ~InvWishart(Q,l3)
% s(t)~N(((W.Z)*s(t-1)),Phi_prime^-1), 
% MCMC initialization 

% model initialization 
if (initilization == "psd")
    init = learn_spectral( K, floor(size(X,2)/2), MM )
    S0 = init.initx;%ones(K,1);
    W = init.A;%((1/K)^2).*ones(K);
    Z = binornd(1,Sparsity_ratio,[K,K]);%ones(K)-eye(K);
    D = init.C;%(1/(K*X_dim_S)).*ones(X_dim_S,K);
    Phi = init.R;
    Phi_prime = init.Q;
    [S, BB, CC, loglik] = kalman_smoother(MM, W.*Z, D, Phi_prime, Phi, S0, init.initV);%ones(K,T);
else
    S = rand(K,T);
    S0 = rand(K,1); %init.initx%ones(K,1);
    W = rand(K);%init.A%((1/K)^2).*ones(K);
    Z = binornd(1,Sparsity_ratio,[K,K]);%ones(K)-eye(K);
    D = rand(X_dim_S,K);%init.C%(1/(K*X_dim_S)).*ones(X_dim_S,K);
    %Phi = init.R;
    %Phi_prime = init.Q;
    %[S, BB, CC, loglik] = kalman_smoother(data(:,1:200), W.*Z, D, Phi_prime, Phi, S0, init.initV);%ones(K,T);
end


% sigma=K^2*eye(K);


theta = rand(K,K_prime);
psi = rand(K_prime,K);
r = rand(K_prime,1);
l1 = randi([0,1],K,K_prime);
l2 = randi([0,1],K_prime,K);
l3 = randi(K_prime,1);
rho = rand(K,1);
tao = rand(K,1);
etta = rand(K_prime,1);
Etta0 = eye(K);
m0 = zeros(K,1);
neu0 = 1;
c = 1;
e=ones(K_prime,1);
f=ones(K_prime,1);
g=1;
Phi_prime= zeros(K);

% initilatation of model hyper parameters 
%Model_hyper_params.a0 = 1;
%Model_hyper_params.b0 = 1;
%Model_hyper_params.c0 = 1;
%Model_hyper_params.d0 = 1;
Model_hyper_params.m00 =1;
Model_hyper_params.n0 = 1;
Model_hyper_params.u0 = 1;
Model_hyper_params.v0 = 1;
Model_hyper_params.w0 = 1;
Model_hyper_params.p0 = 1;
Model_hyper_params.h0 = 1;
Model_hyper_params.g0 = 1;
Model_hyper_params.l0 = 1;
Model_hyper_params.V1 = eye(X_dim_S);
Model_hyper_params.df1 = X_dim_S+2;
Model_hyper_params.V2 = eye(X_dim_S);
Model_hyper_params.df2 = X_dim_S+2;
Model_hyper_params.Etta0 = eye(K);
Model_hyper_params.m0 = zeros(K,1);

Model_hyper_params.alpha0 = 1;
Model_hyper_params.beta0 = 1;

%Model_hyper_params.alpha2 = 1;
%Model_hyper_params.beta2 = 1e-1;
%Model_hyper_params.etta0 = 1;
Model_hyper_params.gamma0 = 1;
Model_hyper_params.neu0 = 1;
Model_hyper_params.ksi0 = 1;
Model_hyper_params.alpha0 = 1;
Model_hyper_params.beta0 = 1;
% Model_hyper_params.alpha1 = 1;
% Model_hyper_params.beta1 = 10.;%1e-2%*ones(K,1);
% Model_hyper_params.alpha2 = 1;
% Model_hyper_params.beta2 = 1e-1;
Model_hyper_params.etta0 = 1;
% Model_hyper_params.gamma0 = 1;
% Model_hyper_params.neu0 = 1;
% Model_hyper_params.ksi0 = 1;


S_avg = zeros(K,T);
D_avg = zeros(X_dim_S,K);
W_avg = zeros(K);
Z_avg = zeros(K);
F_avg = zeros(K);
rankZ_avg = 0;

% creating truncated poisson  
pd = makedist('Poisson');
t_pos = truncate(pd,1,inf);
%F = W.*Z;
kk = 1;
% MCMC Sampling 
for iteration = 1:Max_MCMC

    % smaple M(i,j)
    m_ikj = sample_M_ikj(Z,theta,r,psi,t_pos,Model_properties);
    lambda_ij = theta*diag(r)*psi;
    [theta,psi,r,l1,l2,l3,rho,tao,e,f,g] = sample_theta_psi_r_rho_tao_etta_e_f_g(m_ikj,theta,rho,e,r,f,g,l2,l3,psi,tao,Model_properties,Model_hyper_params);
    
    %sample sigma(i,j)
    sigma = sample_sigma(Model_hyper_params,W,Model_properties);
    % sample phi_prime
    Phi_prime = sample_Phi_prime(Model_hyper_params,T,S,W,Z,S0,Model_properties);
    % sample Phi
    Phi = sample_Phi(Model_hyper_params,X_dim_S,N,X,S,D,Model_properties);


    %sample Lambda(k)
    for k =1:K
       par1 = D(:,k)'*D(:,k)+Model_hyper_params.V2;
       par2 = Model_hyper_params.df2+1;
       temp = iwishrnd(par1,par2);
       Lambda(:,:,k)= temp\eye(X_dim_S);
    end

    % smaple Wij and Zij
  
    %
    [W,Z] = sample_WZ(S0,S,Z,W,Phi_prime,sigma,theta,r,psi,Model_hyper_params,Model_properties);
    [D,S,S0] = sample_SD(X,W,Z,Phi_prime,S0,S,D,Lambda,Phi,Model_hyper_params,Model_properties);




    if (iteration >Burnin)
        S_avg = S_avg+S;
        D_avg = D_avg+D;
        W_avg = W_avg+W;
        Z_avg = Z_avg+Z;
        F_avg = F_avg+(W.*Z);
        rankZ_avg = rankZ_avg+rank(Z);

    end

    if (iteration >Burnin && rem(iteration,60)==0)
        r_col(:,kk) =r;
        m_col(:,:,:,kk) = m_ikj;
        theta_col(:,:,kk) = theta;
        psi_col(:,:,kk)=psi;
        S0_col(:,kk) = S0;
        S_col(:,:,kk) = S;
        D_col(:,:,kk) = D;
        W_col(:,:,kk) = W;
        Z_col(:,:,kk) = Z;
        Sparsity(kk)= nnz(Z)/K^2;
        Matrix_rank(kk) = rank( W.*Z);
        Eig_values(:,kk) = abs(eig(W.*Z));
        Phi_col(:,:,kk) = Phi;
        Phi_prime_col(:,:,kk) = Phi_prime;
        kk = kk+1;
    end
    if rem(iteration,50) ==0 
        fprintf('the Iteration: %d\n', iteration);
        rank(Z.*W)
    end


end
S_avg = S_avg/(Max_MCMC-Burnin);
D_avg = D_avg/(Max_MCMC-Burnin);
W_avg = W_avg/(Max_MCMC-Burnin);
Z_avg = Z_avg/(Max_MCMC-Burnin);
F_avg = F_avg/(Max_MCMC-Burnin);
rankZ_avg = rankZ_avg/(Max_MCMC-Burnin);
for i = 1:size(S_col,3)
   X_recon(:,:,i)= D_col(:,:,i)*S_col(:,:,i);
end
X_recon_avg = mean(X_recon,3);
for t = 1:T
    X_S_avg(:,t) = mean(X{t},2);
end
filename = sprintf('%s%s%s %f %s %d.txt',Folder_name,'/',File_name,portion,'_K',K); 
filename_csv = sprintf('%s%s%s %f %s %d.csv',Folder_name,'/',File_name,portion,'_K',K);
fileID = fopen(filename,'wt');
filename_csv_global = sprintf('%s%s%s %f %s %d.csv',Folder_name,'/',File_name,portion,'_K',K);
%Prediction_K_step(filename,)

try 
    Existing_results = csvread(filename_csv_global);
catch
    Existing_results =[];
end
fileID_csv = fopen(filename_csv,'wt')
fprintf(fileID,'||X||(2)= %f\n',norm(X_S_avg))
row =1; results_csv(row) = norm(X_S_avg,'fro'); row =row+1;
fprintf(fileID,'||X||(F)= %f\n',results_csv(row-1));
fprintf(fileID,'||X - D*S(training)||(2)= %f\n',norm(X_S_avg-X_recon_avg));
results_csv(row) = norm(X_S_avg-X_recon_avg,'fro');row =row+1;
fprintf(fileID,'||X - D*S(training)||(F)= %f\n',results_csv(row-1));
%fprintf('rank(W_syntethic.*Z_syntethic)= %f\n',rank(W_S.*Z_S))
fprintf(fileID,'rank(W_syntethic.*Z__inferred)= %f\n',rankZ_avg);
rank(Z.*W)

% 
% for i=1:length(X_w)
%     Nw(i) = size(X_w{i},2);
% end
% 
% prdiction model initialization 
for t = 1:T_P
    X_W_avg(:,t) = mean(X_w{t},2);
end
for i = 1:size(X,2)
   TRdata(:,i)= X{i}(:,1); 
end
for i = 1:size(X_w,2)
   TEdata(:,i)= mean(X_w{i},2);
end
initV = cov(S0_col');

if strcmp(task,'predeiction')
   
    [XP1,abser1,XP2] = Prediction_Ksteps(X,X_w,T_P,X_dim_S,S0_col,S_col,Phi_prime_col,K,W_col,Z_col,Phi_col,D_col,X_W_avg,T);

    pred_error1 = abser1(:,1:10,:);
    M_pred = mean(abs(X_W_avg(:,1:10)));
    mean_error1 = mean(mean(pred_error1,3),1);
    temp = permute(pred_error1,[1 3 2]);
    temp = reshape(temp,[],size(pred_error1,2),1);
    std_error1 = sqrt(var(temp));

    X_reconp_avg = mean(XP1,3);

    fprintf(fileID,'Average Sparsity level =  %f\n',mean(Sparsity))
    results_csv(row) = mean(Sparsity);row =row+1;
    fprintf(fileID,'Average Rank =  %f\n',mean(Matrix_rank))
    results_csv(row) = mean(Matrix_rank);row =row+1;
    fprintf(fileID,'||X||(2)= %f\n',norm(X_W_avg(:,1:10)+MMM))
    results_csv(row) = norm(X_W_avg(:,1:10)+MMM,'fro');row =row+1;
    fprintf(fileID,'||X||(F)= %f\n',results_csv(row-1))
    results_csv(row) = norm(X_W_avg(:,1:10)-X_reconp_avg(:,1:10),'fro');row =row+1;
    fprintf(fileID,'||X - X_prediction_MCchain_Tsteps||(2)= %f\n',norm(X_W_avg(:,1:10)-X_reconp_avg(:,1:10)))
    fprintf(fileID,'||X - X_prediction_MCchain_Tsteps||(F)= %f\n',results_csv(row-1))
    results_csv(row) = sum(sum(abs(1- (X_reconp_avg(:,1:10)+MMM)./(X_W_avg(:,1:10)+MMM))))/(size(X_reconp_avg(:,1:10),1)*size(X_reconp_avg(:,1:10),2))*100;row =row+1;
    fprintf(fileID,'MAPE_%d_Step_MCchain = %f\n',T_P,results_csv(row-1))
    figure()
    for i = 1:size(X_W_avg(:,1:10),1)
        subplot(ceil(size(X_W_avg(:,1:10),1)/3),ceil(size(X_W_avg(:,1:10),1)/ceil(size(X_W_avg(:,1:10),1)/3)),i)
        plot(X_reconp_avg(i,1:10)+MMM(i))
        hold on 
        plot(X_W_avg(i,1:10)+MMM(i))
        hold off
        title(sprintf('observation dimension %d ',i));
        legend('Predicted_','Real');
    end 
    saveas(gcf,sprintf('%s%s%s%s %f %s %d.fig',Folder_name,'/',File_name,'_r0_data_figure_T_step_prediction1_training size_',portion,'_K',K))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5


    X_reconp_avg2 = mean(XP2,3);

    fprintf(fileID,'||X||(2)= %f\n',norm(X_W_avg(:,1:10)+MMM))
    fprintf(fileID,'||X||(F)= %f\n',norm(X_W_avg(:,1:10)+MMM,'fro'))
    fprintf(fileID,'||X - X_prediction_Kalman_Tsteps||(2)= %f\n',norm(X_W_avg(:,1:10)-X_reconp_avg2(:,1:10)))
    results_csv(row) = norm(X_W_avg(:,1:10)-X_reconp_avg2(:,1:10),'fro');row =row+1;
    fprintf(fileID,'||X - X_prediction_Kalman_Tsteps||(F)= %f\n',results_csv(row-1))
    results_csv(row) = sum(sum(abs(1- (X_reconp_avg2(:,1:10)+MMM)./(X_W_avg(:,1:10)+MMM))))/(size(X_reconp_avg2(:,1:10),1)*size(X_reconp_avg2(:,1:10),2))*100;row =row+1;
    fprintf(fileID,'MAPE_%d_Step_Kalman = %f\n',T_P,results_csv(row-1))
    figure();
    for i = 1:size(X_W_avg,1)
        subplot(ceil(size(X_W_avg(:,1:10),1)/3),ceil(size(X_W_avg(:,1:10),1)/ceil(size(X_W_avg(:,1:10),1)/3)),i)
        plot(X_reconp_avg2(i,1:10)+MMM(i))
        hold on 
        plot(X_W_avg(i,1:10)+MMM(i))
        hold off
        title(sprintf('observation dimension %d ',i));
        legend('Predicted_','Real')
    end
    saveas(gcf,sprintf('%s%s%s%s %f %s %d.fig',Folder_name,'/',File_name,'_r0_figure_T_step_prediction1_kalman_training size_',portion,'_K',K))

    [XP3,abser3] = Prediction_1step(X,X_w,T_P,X_dim_S,S0_col,S_col,Phi_prime_col,K,W_col,Z_col,Phi_col,D_col,X_W_avg,T);

    pred_error3 = abser3(:,1:10,:);
    M_pred = mean(abs(X_W_avg(:,1:10)));
    mean_error3 = mean(mean(pred_error3,3),1);
    temp = permute(pred_error3,[1 3 2]);
    temp = reshape(temp,[],size(pred_error3,2),1);
    std_error3 = sqrt(var(temp));


    X_reconp_avg3 = mean(XP3,3);


    fprintf(fileID,'||X||(2)= %f\n',norm(X_W_avg(:,1:10)+MMM))
    fprintf(fileID,'||X||(1)= %f\n',norm(X_W_avg(:,1:10)+MMM,'fro'))
    fprintf(fileID,'||X - X_prediction_Kalman_1steps||(2)= %f\n',norm(X_W_avg(:,1:10)-X_reconp_avg3(:,1:10)))
    results_csv(row) = norm(X_W_avg(:,1:10)-X_reconp_avg3(:,1:10),'fro');row =row+1;
    fprintf(fileID,'||X - X_prediction_Kalman_1steps||(F)= %f\n',results_csv(row-1))
    results_csv(row) = sum(sum(abs(1- (X_reconp_avg3(:,1:10)+MMM)./(X_W_avg(:,1:10)+MMM))))/(size(X_reconp_avg3(:,1:10),1)*size(X_reconp_avg3(:,1:10),2))*100; row =row+1;
    fprintf(fileID,'MAPE_%d_Step = %f\n',1,results_csv(row-1))
    figure()
    for i = 1:size(X_W_avg,1)
        subplot(ceil(size(X_W_avg(:,1:10),1)/3),ceil(size(X_W_avg(:,1:10),1)/ceil(size(X_W_avg(:,1:10),1)/3)),i)
        plot(X_reconp_avg3(i,1:10)+MMM(i))
        hold on 
        plot(X_W_avg(i,1:10)+MMM(i))
        hold off
        title(sprintf('observation dimension %d ',i));
        legend('Predicted','Real')
    end
    saveas(gcf,sprintf('%s%s%s%s %f %s %d.fig',Folder_name,'/',File_name,'_r0_figure_1_step_prediction_training size_',portion,'_K',K))
elseif strcmp(task,'interpretation')
    for t = 1:T_S
        X_orig_avg1(:,t) = mean(X{t},2);
    end
    [S, V, VV, loglik] = kalman_filter(TRdata, W.*Z, D, inv(Phi_prime), inv(Phi), S0, initV);
    Model_interpretation(m_ikj,r,theta,psi,K_prime,X_orig_avg1,S0,S,K,W,Z,D,T,dataset_name)
else
    fprintf('task is not recognized\n');
    quit
end
if  strcmp(data_reconstruction_flag,'yes')
     X_recon_avg1 = data_reconstruction(X,X_w,S0_col,S_col,Phi_prime_col,W_col,Z_col,Phi_col,D_col)

    fprintf(fileID,'||X||(1)= %f\n',norm(X_orig_avg1+MMM,'fro'))
    fprintf(fileID,'||X - X_original_recon_kalman||(2)= %f\n',norm(X_orig_avg1-X_recon_avg1))
    results_csv(row) = norm(X_orig_avg1-X_recon_avg1,'fro');row =row+1
    fprintf(fileID,'||X - X_original_recon_kalman||(F)= %f\n',results_csv(row-1))
    results_csv(row) = sum(sum(abs(1- (X_recon_avg1+MMM)./(X_orig_avg1+MMM))))/(size(X_recon_avg1,1)*size(X_recon_avg1,2))*100;row =row+1;
    fprintf(fileID,'MAPE_%d_Step = %f\n',1,results_csv(row-1))

    figure()
    for i = 1:size(X_orig_avg1,1)
        subplot(ceil(size(X_orig_avg1,1)/3),ceil(size(X_orig_avg1,1)/ceil(size(X_orig_avg1,1)/3)),i)
        plot(X_recon_avg1(i,:)+MMM(i))
        hold on 
        plot(X_orig_avg1(i,:)+MMM(i))
        hold off
        title(sprintf('observation dimension %d ',i));
        legend('Predicted_','Real')
    end
    saveas(gcf,sprintf('%s%s%s %f %s %d.fig',Folder_name,'/',File_name,portion,'_K',K))
end
csvwrite(filename_csv_global, [Existing_results results_csv']);
csvwrite(filename_csv,results_csv',1,run_number);
filename_workspace =  sprintf('%s%s%s %f %s %d',Folder_name,'/',File_name,portion,'_K',K,'.mat')
save(filename_workspace)
fclose('all');
%end

