clc
clear; close all; fclose all;
path(path,'.\sample_toolbox\') ;
%  load  data
dataname = 'Covid19_newcases'% 
%chose the task
TypeofEvent = 'concant_all' ; %'cases' , 'death', 'concant_all'
task = 'prediction_historic';       % 'prediction_future', 'prediction_historic'



%chose the task
Folder_name = 'results';


switch TypeofEvent                               
	case   'death'
        TT = readtable('data/new_death_cases.csv');
        data =TT(1:end,[1,55:size(TT,2)]);
        data1 =TT{2:end,55:end};
        data6=[];
        data6=[data6; cellfun(@str2num, data1(:,1:end))];
        data_all=data6;
        data{2:end,2:end}=num2cell(data6);
        data(end+1,1)=cellstr('US');
        data{end,2:end}=num2cell(sum(data6,1));
        V = size(data_all,1);  
        T_initial = 2;            % 3/15/2020
	case   'cases'
        TT = readtable('data/new_daily_cases.csv');
        data =TT(1:end,[1,55:size(TT,2)]);
        data1 =TT{2:end,55:end};
        data6=[];
        data6=[data6; cellfun(@str2num, data1(:,1:end))];
        data_all=data6;
        data{2:end,2:end}=num2cell(data6);
        data(end+1,1)=cellstr('US');
        data{end,2:end}=num2cell(sum(data6,1));
        V = size(data_all,1);  
        T_initial = 2;            % 3/15/2020
	case   'concant_all'
        TT1 = readtable('data/new_death_cases.csv');
        TT2 = readtable('data/new_daily_cases.csv');
        data1 =TT1(1:end,[1,55:size(TT1,2)]);
        data2 =TT1{2:end,55:end};
        data3 =TT2(1:end,[1,55:size(TT2,2)]);
        data4 =TT2{2:end,55:end};
        data7=[];
        data7=[data7; cellfun(@str2num, data2(:,1:end))];
        data6=[];
        data6=[data6; cellfun(@str2num, data4(:,1:end))];
        data_all= [data7;data6];

        data1{2:end,2:end}=num2cell(data7);
        data1(end+1,1)=cellstr('US');
        data1{end,2:end}=num2cell(sum(data7,1));  % new_death_cases
        
        data3{2:end,2:end}=num2cell(data6);
        data3(end+1,1)=cellstr('US');
        data3{end,2:end}=num2cell(sum(data6,1));  % new_daily_cases
        V = size(data_all,1);  
        T_initial = 2;            % 3/15/2020

end 
%%  processing the data
if strcmp(task,'prediction_historic')
    Folder_name1 = 'Historic_prediction';
    T         = size(data_all,2)-7;
    T_test    =    7;  
elseif strcmp(task,'prediction_future')
    Folder_name1 = 'Future_prediction';
    T          = size(data_all,2);
    T_test       = 7;
end

X_test=[];    
for i = 1:size(data_all,2)
    if i <= T
        X_train(:,i) = double(data_all(:,i));
    else
        X_test(:,i-T) = double(data_all(:,i));
    end
end

TRdata = X_train;
TEdata = X_test;
TrX = cat(2, TRdata, TEdata);
%%  Setting 
K =   [40];
L = length(K);
Setting.num_trial     = 1;
Setting.Burnin        =   1500;
Setting.Collection    =   2500;
Setting.Stationary    =   0;
Setting.Sample_interval       =   50;
Setting.step          = 10;
Setting.Pred_timestep = T_test;
Setting.pred_count     = 100;
%%  save the figure and the predicted results
if  L==1
    save_file = ['./results/', Folder_name1,'/','layer',num2str(L),'/','K1_',num2str(K(1)),'/'];
elseif L==2 
    save_file = ['./results/', Folder_name1,'/','layer',num2str(L),'/','K1_',num2str(K(1)),'_K2_',num2str(K(2)),'/'];
else
    save_file = ['./results/', Folder_name1,'/','layer',num2str(L),'/','K1_',num2str(K(1)),'_K2_',num2str(K(2)),'_K3_',num2str(K(3)),'/'];
end

if ~ exist(save_file)
mkdir(save_file);
end

%%   our model
for trial =1:Setting.num_trial
       kk=1;

 %       X_pred{trial}    =  zeros(Setting.Collection,Setting.num_chain,V,Setting.Pred_timestep);
%        rec_data{trial}  =  zeros(V,T);

        % Set hyper-parameters
        Supara.tao0 = 1;
        Supara.gamma0 = 100;
        Supara.eta0 = 0.1;
        Supara.epilson0 = 0.1;
        Supara.c = 1;
        % Initialise the global and local parameters 
        Para.Phi = cell(L,1);
        Para.Pi  = cell(L,1);
        Para.Xi  = cell(L,1);
        Para.V  = cell(L,1);
        Para.beta  = cell(L,1);
        Para.q  = cell(L,1);
        Para.h = cell(L,1);
        Para.n = cell(L,1);
        Para.rou = cell(L,1);       
        Piprior = cell(L,1);
        Theta    = cell(L,1);
        delta  = cell(L,1);
        Zeta  = cell(L,1);
        L_dotkt  = cell(L,1);
        L_kdott = cell(L,1);
        A_KT = cell(L,1);
        A_VK = cell(L,1);
        
        L_KK = cell(L,1);
        prob1 = cell(L,1);
        prob2 = cell(L,1);
        Xt_to_t1 = cell(L+1,1);
        X_layer_split1 = cell(L,1);
        X_layer  = cell(L,1);
        
        for l=1:L
            if l==1
                Para.Phi{l} = rand(V,K(l));
                A_VK{l} = zeros(V,K(l));
            else
                Para.Phi{l} = rand(K(l-1),K(l));
                A_VK{l} = zeros(K(l-1),K(l));
            end
            Para.Phi{l} = bsxfun(@rdivide, Para.Phi{l}, max(realmin,sum(Para.Phi{l},1)));
            Para.Pi{l}  = eye(K(l));
            Para.Xi{l} = 1;
            Para.V{l} = ones(K(l),1);
            Para.beta{l} = 1;
            Para.h{l} = zeros(K(l),K(l));
            Para.n{l} = zeros(K(l),1);
            Para.rou{l} = zeros(K(l),1);
            
            Theta{l}    = ones(K(l),T)/K(l);
            delta{l}    = ones(T,1);
            Zeta{l}     = zeros(T+1,1);
            L_dotkt{l} = zeros(K(l),T+1);
            A_KT{l} = zeros(K(l),T);
            L_kdott{l} = zeros(K(l),T+1);
            X_layer{l} = zeros(K(l),T,2);
            
        end        
%%  ---------------------------------Traning the model-----------------------------------------------               
        for i = 1:Setting.Collection+Setting.Burnin
            tic
           for l=1:L
    
                L_KK{l} = zeros(K(l),K(l));  
                Thetatmp =  bsxfun(@times,delta{1}', Theta{1});
                if l==1
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(X_train), Para.Phi{l},Thetatmp); 
                else                
                    [A_KT{l},A_VK{l}] = Multrnd_Matrix_mex_fast_v1(sparse(Xt_to_t1{l}), Para.Phi{l},Thetatmp);                
                end                         
                  %% sample next layer count 
                  if l == L            

                        for t=T:-1:2
                            L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KT{l}(:,t)+ L_dotkt{l}(:,t+1))'),(Supara.tao0 * Para.Pi{l} * Theta{l}(:,t-1))')';
                            [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott{l}(:,t)), Para.Pi{l},Theta{l}(:,t-1));
                            L_KK{l} = L_KK{l} + tmp;
                        end      

                  else    

                        prob1{l} = Supara.tao0 * Para.Pi{l} *  Theta{l};         
                        prob2{l} = Supara.tao0 * Para.Phi{l+1} *  Theta{l+1}; 
                        X_layer{l} = zeros(K(l),T,2);
                        Xt_to_t1{l+1} = zeros(K(l),T);
                        X_layer_split1{l} = zeros(K(l),T);

                        for t = T : -1 : 2               
                             L_kdott{l}(:,t) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,t)'+ L_dotkt{l}(:,t+1)'),Supara.tao0*( Para.Phi{l+1}*Theta{l+1}(:,t)+ Para.Pi{l} * Theta{l}(:,t-1))')';      
                            %% split layer2 count            
                             [~,X_layer{l}(:,t,:)] = Multrnd_Matrix_mex_fast_v2(L_kdott{l}(:,t),[prob1{l}(:,t-1) prob2{l}(:,t)],ones(2,1));
                             X_layer_split1{l}(:,t) = squeeze(X_layer{l}(:,t,1));   %pi1*Theta1
                             Xt_to_t1{l+1}(:,t) = squeeze(X_layer{l}(:,t,2));   %phi2*Theta2
                            %% sample split1 augmentation
                             [L_dotkt{l}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(X_layer_split1{l}(:,t)), Para.Pi{l},Theta{l}(:,t-1));
                             L_KK{l} = L_KK{l} + tmp ;             
                        end

                        L_kdott{l}(:,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{l}(:,1)'+ L_dotkt{l}(:,2)'),Supara.tao0*( Para.Phi{l+1}*Theta{l+1}(:,1))')';    
                        Xt_to_t1{l+1}(:,1) = L_kdott{l}(:,1); 

                  end
           
              
            %% sample Phi  
                Para.Phi{l} = SamplePhi(A_VK{l},Supara.eta0);
                if nnz(isnan(Para.Phi{l}))
                    warning('Phi Nan');
                    Para.Phi{l}(isnan(Para.Phi{l})) = 0;
                end
            
            %% sample Pi 
                Piprior{l} = Para.V{l}*Para.V{l}';
                Piprior{l}(logical(eye(size(Piprior{l}))))=0;
                Piprior{l} = Piprior{l}+diag(Para.Xi{l}*Para.V{l});
                Para.Pi{l} = SamplePi(L_KK{l},Piprior{l});
                if nnz(isnan(Para.Pi{l}))
                    warning('Pi Nan');
                    Para.Pi{l}(isnan(Para.Pi{l})) = 0;
                end
                
            end
            %% calculate Zeta  
            if Setting.Stationary == 1
                for l=1:L
                    if l==1
                        Zeta{l}= -lambertw(-1,-exp(-1-delta{l}./Supara.tao0))-1-delta{l}./Supara.tao0;
                    else
                        Zeta{l} = -lambertw(-1,-exp(-1-Zeta{l-1}))-1-Zeta{l-1};
                    end
                    Zeta{l}(T+1) = Zeta{l}(1);
                    L_dotkt{l}(:,T+1) = poissrnd(Zeta{l}(1)*Supara.tao0 * Theta{l}(:,T));
                end
            end
            
            
            if Setting.Stationary == 0
                for l=1:L
                    if l==1
                        for t=T:-1:1
                            Zeta{l}(t) = log(1 + delta{l}(t)/Supara.tao0 + Zeta{l}(t+1));
                        end
                    else
                        for t=T:-1:1
                            Zeta{l}(t) = Supara.tao0*log(1+Zeta{l-1}(t)+Zeta{l}(t+1));
                        end
                        
                    end
                    
                end
            end
            %% sample Theta 
        for l=L:-1:1
           
           if l==L
                for t=1:T
                    if t==1
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.V{l};
                    else
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0*(Para.Pi{l}* Theta{l}(:,t-1));
                    end
                    scale = Supara.tao0 + delta{l}(t)+ Zeta{l}(t+1);
                    Theta{l}(:,t) = gamrnd(shape,1./scale);
                 end 
         
           else
                for t=1:T
                    if t==1
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * Para.Phi{l+1}* Theta{l+1}(:,t);
                    else
                        shape = A_KT{l}(:,t)+ L_dotkt{l}(:,t+1)+ Supara.tao0 * (Para.Phi{l+1} * Theta{l+1}(:,t)+ Para.Pi{l}* Theta{l}(:,t-1));
                    end
                        scale = ( delta{l}(t) + Supara.tao0 + Supara.tao0 * Zeta{l}(t+1))';                
                    Theta{l}(:,t) = gamrnd(shape,1./scale);
                end
               
               
           end
           
                       if nnz(isnan(Theta{l}))
                            warning('Theta Nan');
                       end
                    
       end          
            
            
           
            %% sample Beta    
            for l = 1:L
                shape = Supara.epilson0 + Supara.gamma0;
                scale = Supara.epilson0 + sum(Para.V{l});
                Para.beta{l} = gamrnd(shape,1./scale);
                %% sample q  
                a  = sum(L_dotkt{l},2);
                b  = Para.V{l}.*(Para.Xi{l}+repmat(sum(Para.V{l}),K(l),1)-Para.V{l});
                Para.q{l} = betarnd(b,a);
                Para.q{l} = max(Para.q{l},realmin);
                %% sample h   
                for k1 = 1:K(l)
                    for k2 = 1:K(l)
                        Para.h{l}(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK{l}(k1,k2)),Piprior{l}(k1,k2));
                    end
                end
                %% sample Xi  
                shape = Supara.gamma0/K(l) + trace(Para.h{l});
                scale = Para.beta{l} - Para.V{l}'*log(Para.q{l});
                Para.Xi{l} = gamrnd(shape,1./scale);           
            end
            %% sample Delta 1  
            [ delta{1} ] =Sample_delta(X_train,Theta{1},Supara.epilson0,Setting.Stationary);
            
            if nnz(isnan(delta{1}))
                warning('delta Nan');
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% sample V{L}
            for k=1:K(L)
                L_kdott{L}(k,1) = CRT_sum_mex_matrix_v1(sparse(A_KT{L}(k,1)+ L_dotkt{L}(k,2)),Supara.tao0*Para.V{L}(k));
                Para.n{L}(k)=sum(Para.h{L}(k,:)+Para.h{L}(:,k)')-Para.h{L}(k,k) + L_kdott{L}(k,1);
                Para.rou{L}(k) = -log(Para.q{L}(k)) * (Para.Xi{L}+sum(Para.V{L})-Para.V{L}(k)) - (log(Para.q{L}')*Para.V{L}-log(Para.q{L}(k))*Para.V{L}(k)) + Zeta{L}(1);
            end
            shape = Supara.gamma0/K(L) + Para.n{L};
            scale = Para.beta{L} + Para.rou{L};
            Para.V{L} = gamrnd(shape,1./scale);
            
            %% sample V{1},...,V{L]
            if L>1
                for l=1:L-1
                    for k=1:K(l)
                        Para.n{l}(k)=sum(Para.h{l}(k,:) + Para.h{l}(:,k)')-Para.h{l}(k,k);
                        Para.rou{l}(k) = -log(Para.q{l}(k)) * (Para.Xi{l}+sum(Para.V{l})-Para.V{l}(k)) - (log(Para.q{l}')*Para.V{l}-log(Para.q{l}(k))*Para.V{l}(k));% + Para.Zeta2(1);
                    end
                    shape = Supara.gamma0/K(l) + Para.n{l};
                    scale = Para.beta{l} + Para.rou{l};
                    Para.V{l} = gamrnd(shape,1./scale);
                    
                    if nnz(isnan( Para.V{l}))
                        warning('V Nan');
                    end
                end
            end
            %%
            Lambda = bsxfun(@times,delta{1}', Para.Phi{1}*Theta{1});
            like   = sum(sum( X_train .* log(Lambda)-Lambda))/V/T;
            Likelihood(i) = like;
            if mod(i,Setting.step)==0
                            time_iter = toc;
                            fprintf('dataname:%s,iter: %d, like: %d  \n',dataname,i,like);
                            fprintf('trial %d,time_iter %f \n',trial,time_iter);
                            fprintf('.................................................................................... \n')                        
            end

       

             if (i >Setting.Burnin && rem(i,Setting.Sample_interval)==0)
                Para_sample{kk}= Para;
                Theta_sample{kk}= Theta;
                delta_sample{kk}=delta;
                kk = kk +1 ;                   
             end
                            
        end
        %%  
        if strcmp(task,'prediction_historic')            
            [XP_temp,abser_death,abser_cases,X_recon] = Predict_multisteps_con(Para_sample,Theta_sample,T,T_test,V,L,Setting.pred_count,delta_sample,TEdata,task);   

       
             pred_error_death = abser_death(:,1:end,:);
             mean_error_death = mean(mean(pred_error_death,3),1);
            temp = permute(pred_error_death,[1 3 2]);
            temp = reshape(temp,[],size(pred_error_death,2),1);
            std_error_death = sqrt(var(temp));             
            %%
             pred_error_cases = abser_cases(:,1:end,:);
             mean_error_cases = mean(mean(pred_error_cases,3),1);
            temp = permute(pred_error_cases,[1 3 2]);
            temp = reshape(temp,[],size(pred_error_cases,2),1);
            std_error_cases = sqrt(var(temp));             
        else            
            TEdata=[];            
            [XP_temp,abser_death,abser_cases,X_recon] = Predict_multisteps_con(Para_sample,Theta_sample,T,T_test,V,L,Setting.pred_count,delta_sample,TEdata,task);   
        end
%%
        XP_temp_death = XP_temp(1:V/2,:,:,:);      X_recon_death  = X_recon(1:V/2,:,:);
        
        X_recon_avg_death   =  mean(X_recon_death,3);      
        X_reconp_avg_death  =  squeeze(mean(mean(XP_temp_death,4),2));
        MyX_death    =  cat(2, X_recon_avg_death, (X_reconp_avg_death));        

        XP_all_death =  cat(1,XP_temp_death,sum(XP_temp_death,1));
        
        US_total_obse_death = sum(data_all(1:V/2,1:T),1);  
        US_avg_death = [US_total_obse_death,sum(X_reconp_avg_death,1)];
        MyX_all_death = cat(1,MyX_death,US_avg_death);
 %%       
        XP_temp_cases = XP_temp(V/2+1:end,:,:,:);  X_recon_cases  = X_recon(V/2+1:end,:,:);
        X_recon_avg_cases  = mean(X_recon_cases,3);      
        X_reconp_avg_cases = squeeze(mean(mean(XP_temp_cases,4),2));
        MyX_cases = cat(2, X_recon_avg_cases, (X_reconp_avg_cases));        

        XP_all_cases =cat(1,XP_temp_cases,sum(XP_temp_cases,1));
        US_total_obse_cases = sum(data_all(V/2+1:end,1:T),1);  
        US_avg_cases = [US_total_obse_cases,sum(X_reconp_avg_cases,1)];
        MyX_all_cases = cat(1,MyX_cases,US_avg_cases);  

   
        TrX_death = cat(1, TrX(1:V/2,:), sum(data_all(1:V/2,:),1));
        TrX_cases = cat(1, TrX(V/2+1:end,:), sum(data_all(V/2+1:end,:),1));
%%         
        
         close all
         
         for i = 1:size(MyX_all_death,1)
            figure(i)
            temp11 = permute(squeeze(XP_all_death(i,:,:,:)),[1 3 2]);
            temp11 = reshape(temp11,[],size(temp11,3),1);
                                    
            xxx=[T+1:T+T_test];
            shadedErrorBar(xxx,temp11,{@mean,@(xxx) CI_values(xxx)},'lineProps',{'b','markerfacecolor','r'})
            temp22 = CI_values(temp11);
            state_upperBound_death(i,:)= temp22(1,:);
            state_lowerBound_death(i,:)= temp22(2,:);

            hold on;
            plot([1:T+T_test], MyX_all_death(i,1:end), 'b');

            if strcmp(task,'prediction_historic')
                plot([1:T+T_test],TrX_death(i,1:end),'r')
            else
                plot([1:T],TrX_death(i,1:end),'r')
            end
            hold on 
            max_y = max(max(TrX_death(i,:)),max(MyX_all_death(i,:)));
            line([T,T],[0,max_y ] ,'Color','black','LineStyle','--')
            hold off
            title(sprintf(string(data1{i+1,1}))); %1-52 death
            
            legend('Predicted1','Predicted2','Real');
            
            fn  =  [save_file  data1{i+1,1}{1} '_death_prediction_' '_layer' num2str(L)  '.jpg'];
            saveas(gcf,fn);
            
         end
        close all
         for i = 1:size(MyX_all_cases,1)
            figure(i)
            temp11 = permute(squeeze(XP_all_cases(i,:,:,:)),[1 3 2]);
            temp11 = reshape(temp11,[],size(temp11,3),1);
                                    
            xxx=[T+1:T+T_test];
            shadedErrorBar(xxx,temp11,{@mean,@(xxx) CI_values(xxx)},'lineProps',{'b','markerfacecolor','r'})
            temp22 = CI_values(temp11);
            state_upperBound_cases(i,:)= temp22(1,:);
            state_lowerBound_cases(i,:)= temp22(2,:);

            hold on;
            plot([1:T+T_test], MyX_all_cases(i,1:end), 'b');

            if strcmp(task,'prediction_historic')
                plot(  [1:T+T_test], TrX_cases(i,1:end),'r')
            else
                plot(     [1:T]    , TrX_cases(i,1:end),'r')
            end
            
            hold on 
            max_y = max(max(TrX_cases(i,:)),max(MyX_all_cases(i,:)));
            line([T,T],[0,max_y ] ,'Color','black','LineStyle','--')
            hold off
     
            title(sprintf(string(data3{i+1,1})));

            
            legend('Predicted1','Predicted2','Real');
            
        
            fn  =  [save_file  data3{i+1,1}{1} '_daily_cases_prediction_' '_layer' num2str(L)  '.jpg'];
            saveas(gcf,fn);
                        
         end
                

       if strcmp(task,'prediction_historic')
            T_means_death = data1;
            T_upperbound_death = data1;
            T_lowerbound_death = data1;
            T_means_death{2:end,T_initial:T_initial+T+T_test-1}=num2cell(MyX_all_death);
            T_lowerbound_death{2:end,T_initial+T:T_initial+T+T_test-1}=num2cell(state_lowerBound_death);
            T_upperbound_death{2:end,T_initial+T:T_initial+T+T_test-1}=num2cell(state_upperBound_death);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            T_means_cases = data3;
            T_upperbound_cases = data3;
            T_lowerbound_cases = data3;
            T_means_cases{2:end,T_initial:T_initial+T+T_test-1}=num2cell(MyX_all_cases);
            T_lowerbound_cases{2:end,T_initial+T:T_initial+T+T_test-1}=num2cell(state_lowerBound_cases);
            T_upperbound_cases{2:end,T_initial+T:T_initial+T+T_test-1}=num2cell(state_upperBound_cases);            
        elseif strcmp(task,'prediction_future')
            T_means_death      =   data1;
            T_upperbound_death =   data1;
            T_lowerbound_death =   data1;
            last_day = datetime(data1{1,end},'Format', 'M/d/yy');
            last_pred_day = daysadd(last_day,7);
            T_means_death{1,T_initial+T:T_initial+T+T_test-1}    =  cellstr(last_day+1:last_pred_day);
            T_means_death{2:end,T_initial:T_initial+T+T_test-1}  =  num2cell(MyX_all_death);

            T_upperbound_death{1,T_initial+T:T_initial+T+T_test-1}     = cellstr(last_day+1:last_pred_day);                        
            T_upperbound_death{2:end,T_initial+T:T_initial+T+T_test-1} = num2cell(state_upperBound_death);            
            T_lowerbound_death{1,T_initial+T:T_initial+T+T_test-1}     = cellstr(last_day+1:last_pred_day);            
            T_lowerbound_death{2:end,T_initial+T:T_initial+T+T_test-1} = num2cell(state_lowerBound_death);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            T_means_cases      =   data3;
            T_upperbound_cases =   data3;
            T_lowerbound_cases =   data3;
            last_day = datetime(data3{1,end},'Format', 'M/d/yy');
            last_pred_day = daysadd(last_day,7);
            T_means_cases{1,T_initial+T:T_initial+T+T_test-1}    =  cellstr(last_day+1:last_pred_day);
            T_means_cases{2:end,T_initial:T_initial+T+T_test-1}  =  num2cell(MyX_all_cases);

            T_upperbound_cases{1,T_initial+T:T_initial+T+T_test-1}     = cellstr(last_day+1:last_pred_day);                        
            T_upperbound_cases{2:end,T_initial+T:T_initial+T+T_test-1} = num2cell(state_upperBound_cases);            
            T_lowerbound_cases{1,T_initial+T:T_initial+T+T_test-1}     = cellstr(last_day+1:last_pred_day);            
            T_lowerbound_cases{2:end,T_initial+T:T_initial+T+T_test-1} = num2cell(state_lowerBound_cases);
        end       
         

        
              
                filename_table = [save_file 'death','_mean_','.csv']
                writetable(T_means_death,filename_table)
                filename_table = [save_file 'death','_lowerBound_','.csv']         
                writetable(T_lowerbound_death,filename_table)
                filename_table = [save_file 'death','_upperBound_','.csv']         
                writetable(T_upperbound_death,filename_table)
                
              
                filename_table = [save_file 'daily_cases','_mean_','.csv']
                writetable(T_means_cases,filename_table)
                filename_table = [save_file 'daily_cases','_lowerBound_','.csv']         
                writetable(T_lowerbound_cases,filename_table)
                filename_table = [save_file 'daily_cases','_upperBound_','.csv']         
                writetable(T_upperbound_cases,filename_table)

        
        name_save = [dataname,'_','_trial',num2str(trial),TypeofEvent,'_S',num2str(Setting.Stationary),'.mat'];
        save([save_file,name_save])                
end


%  load data and calculate the predicted error
% file_name = '.\Covid_PMSE_result_layer2_update_xi_v_delta_k1_15_k2_10_k3_5\';
% name_save = 'Covid__trial7MARE_Layer3_S0.mat';
% load([file_name,name_save])

% predict_mean= mean(estimate_error_meandimension,1);
% predict_std = std(estimate_error_meandimension,1);
% 
% % plot reconstruction samples
% State = {'HI','AZ','IA','MN','OH','WV'};
% INDEX = [13,   4,   17,  25,  37,  50]-1;
% close all;
% 
% for fig_index =1: length(INDEX)
% figure(fig_index);plot(X_train(INDEX(fig_index),2:end),'r','linewidth',2);hold on;
% plot(poissrnd(Lambda(INDEX(fig_index),2:end)),'g','linewidth',2);hold on;
% legend('ground truth','reconstruction');
% set(gca,'FontSize',24);
% title(State{fig_index});
% end
