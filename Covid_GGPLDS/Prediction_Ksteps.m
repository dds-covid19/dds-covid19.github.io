function [XP1,abser1,XP2] = Prediction_Ksteps(X,X_w,T_P,X_dim_S,S0_col,S_col,Phi_prime_col,K,W_col,Z_col,Phi_col,D_col,X_W_avg,T)



    for i=1:length(X_w)
        Nw(i) = size(X_w{i},2);
    end

    % prdiction model initialization 

    pred_count = 100;
    X_pred = zeros(X_dim_S,T_P);
    for i = 1:size(S_col,3)
        for t = 1:size(X_w,2)
        %     if t == 1
        %         SP_temp = FP * S_avg(:,T);
        %         XP(:,t) = DP * SP_temp;
        %     else
        %         SP_temp = FP * SP_prev;
        %         XP(:,t) = DP * SP_temp;
        %     end
        %     SP_prev = SP_temp;

            if t == 1
                temp_ch = chol(Phi_prime_col(:,:,i));
                inv_temp = temp_ch\eye(K);
                z = mvnrnd(zeros(K,1),eye(K),pred_count);
                SP_temp = inv_temp*z' + (W_col(:,:,i).*Z_col(:,:,i))*S_col(:,T,i);        
                %SP_temp = mvnrnd(((W_col(:,:,i).*Z_col(:,:,i))*S_col(:,T,i)),inv(Phi_prime_col(:,:,i)),pred_count);
                temp_ch = chol(Phi_col(:,:,i));
                inv_temp = temp_ch\eye(X_dim_S);
                for j=1:pred_count
                    z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S));
                    XP_temp(:,j) =  inv_temp*z'+ D_col(:,:,i)*SP_temp(:,j);
                end
                %XP_temp1 = mvnrnd((D_col(:,:,i)*SP_temp')',inv(Phi_col(:,:,i)))';
                XP1(:,t,i) = mean(XP_temp,2);
                abser1(:,t,i) = abs(XP1(:,t,i)- X_W_avg(:,t));
            else
                temp_ch = chol(Phi_prime_col(:,:,i));
                inv_temp = temp_ch\eye(K);
                temp_ch1 = chol(Phi_col(:,:,i));
                inv_temp1 = temp_ch1\eye(X_dim_S);
                for j=1:pred_count
                    z = mvnrnd(zeros(K,1),eye(K));
                    SP_temp(:,j) = inv_temp*z' + (W_col(:,:,i).*Z_col(:,:,i))*SP_prev(:,j);
                    z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S));
                    XP_temp(:,j) =  inv_temp1*z'+ D_col(:,:,i)*SP_temp(:,j);
                end
    %                 for j=1:pred_count
    %                     SP_temp(j,:) = mvnrnd(((W_col(:,:,i).*Z_col(:,:,i))*SP_prev(j,:)')',inv(Phi_prime_col(:,:,i)));
    %                     XP_temp = mvnrnd((D_col(:,:,i)*SP_temp(j,:)')',inv(Phi_col(:,:,i)))';
    %                 end
    %                 SP_temp1 = mvnrnd(((W_col(:,:,i).*Z_col(:,:,i))*SP_prev')',inv(Phi_prime_col(:,:,i)));
    %                 XP_temp1 = mvnrnd((D_col(:,:,i)*SP_temp')',inv(Phi_col(:,:,i)))';
                XP1(:,t,i) = mean(XP_temp,2);
                abser1(:,t,i) = abs(XP1(:,t,i)- X_W_avg(:,t));

            end
            SP_prev = SP_temp;
        end
        %X_pred(:,:,i) = 
    end



    for i = 1:size(X,2)
       TRdata(:,i)= X{i}(:,1); 
    end
    for i = 1:size(X_w,2)
       TEdata(:,i)= mean(X_w{i},2);
    end
    initV = cov(S0_col');

    % pred_count = 100;
    % X_pred = zeros(X_dim_S,T_P);
    % for i = 1:size(S_col,3)
    %     for t = 1:size(X_w,2)
    %         if t == 1
    %             [S, V, VV, loglik] = kalman_filter(TRdata(:,T_P-25+t:T_P) , W_col(:,:,i).*Z_col(:,:,i), D_col(:,:,i), inv(Phi_prime_col(:,:,i)), inv(Phi_col(:,:,i)), S_col(:,T_P-25+t-1,i), cov(S_col(:,T_P-25+t-1,i)'));      
    %             %SP_temp = mvnrnd(((W_col(:,:,i).*Z_col(:,:,i))*S_col(:,T,i)),inv(Phi_prime_col(:,:,i)),pred_count);
    %             temp_ch = chol(Phi_col(:,:,i));
    %             inv_temp = temp_ch\eye(X_dim_S);
    %             z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S),pred_count);
    %             XP_temp =  inv_temp*z'+ D_col(:,:,i)*S(:,end);
    %             %XP_temp1 = mvnrnd((D_col(:,:,i)*SP_temp')',inv(Phi_col(:,:,i)))';
    %             XP5(:,t,i) = mean(XP_temp,2);
    %             S_col_temp{i} = [S_col(:,:,i) S(:,end)];
    %         else
    %             [S, V, VV, loglik] = kalman_filter([TRdata(:,T_P-25+t:T_P) XP1(:,1:t-1,i)], W_col(:,:,i).*Z_col(:,:,i), D_col(:,:,i), inv(Phi_prime_col(:,:,i)), inv(Phi_col(:,:,i)), S_col_temp{i}(:,T_P-25+t-1), cov(S_col_temp{i}(:,T_P-25+t-1)'));
    %             temp_ch1 = chol(Phi_col(:,:,i));
    %             inv_temp1 = temp_ch1\eye(X_dim_S);
    %             z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S),pred_count);
    %             XP_temp =  inv_temp1*z'+ D_col(:,:,i)*S(:,end);
    %             XP5(:,t,i) = mean(XP_temp,2);
    %             S_col_temp{i}=[S_col_temp{i} S(:,end)];
    %         end
    %        % S_col_copy(:,:,i) = SP_temp;
    %     end
    %     %X_pred(:,:,i) = 
    % end
    % X_reconp_avg5 = mean(XP5,3);
    % figure()
    % for i = 1:size(X_W_avg,1)
    %     subplot(ceil(size(X_W_avg,1)/3),ceil(size(X_W_avg,1)/ceil(size(X_W_avg,1)/3)),i)
    %     plot(X_reconp_avg5(i,:)+MMM(i))
    %     hold on 
    %     plot(X_W_avg(i,:)+MMM(i))
    %     hold off
    %     title(sprintf('observation dimension %d ',i));
    %     legend('Predicted_','Real');
    % end 
    % saveas(gcf,sprintf('%s%s%s%s %f %s %d%s%d.fig',Folder_name,'/',File_name,'_r0_figure_T_step_prediction_1_step_kalman_training size_',portion,'_K',K,'_iteration_',run_number))
    % 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    for i = 1:size(S_col,3)
        [S, V, VV, loglik] = kalman_filter(TRdata, W_col(:,:,i).*Z_col(:,:,i), D_col(:,:,i), inv(Phi_prime_col(:,:,i)), inv(Phi_col(:,:,i)), S0_col(:,i), initV);
        for t = 1:size(X_w,2)  
            if t == 1
                S_est(:,t) = (W_col(:,:,i).*Z_col(:,:,i))*S(:,end); 
                XP_temp = (D_col(:,:,i)*S_est(:,t));
                XP2(:,t,i) = mean(XP_temp,2);
            else
                S_est(:,t) = (W_col(:,:,i).*Z_col(:,:,i))*S_est(:,t-1); 
                XP_temp = (D_col(:,:,i)*S_est(:,t));
                XP2(:,t,i) = mean(XP_temp,2);
            end
        end
    end
end


