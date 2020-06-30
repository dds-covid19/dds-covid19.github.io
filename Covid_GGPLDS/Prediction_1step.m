function [XP3,abser3] = Prediction_1step(X,X_w,T_P,X_dim_S,S0_col,S_col,Phi_prime_col,K,W_col,Z_col,Phi_col,D_col,X_W_avg,T)

    for i = 1:size(X,2)
       TRdata(:,i)= X{i}(:,1); 
    end
    for i = 1:size(X_w,2)
       TEdata(:,i)= mean(X_w{i},2);
    end
    initV = cov(S0_col');
    for i = 1:size(S_col,3)
        for t = 1:size(X_w,2) 
            [S, V, VV, loglik] = kalman_filter([TRdata TEdata(:,1:t-1)], W_col(:,:,i).*Z_col(:,:,i), D_col(:,:,i), inv(Phi_prime_col(:,:,i)), inv(Phi_col(:,:,i)), S0_col(:,i), initV);
            if t == 1
                SP_temp = (W_col(:,:,i).*Z_col(:,:,i))*S(:,end); 
                XP_temp = (D_col(:,:,i)*SP_temp);
                XP3(:,t,i) = mean(XP_temp,2);
                abser3(:,t,i) = abs(XP3(:,t,i)- X_W_avg(:,t));
            else

                SP_temp = (W_col(:,:,i).*Z_col(:,:,i))*S(:,end);
                XP_temp = (D_col(:,:,i)*SP_temp);
                %XP_temp = mvnrnd((D_col(:,:,i)*SP_temp)',inv(Phi_col(:,:,i)),pred_count)';
                XP3(:,t,i) = mean(XP_temp,2);
                abser3(:,t,i) = abs(XP3(:,t,i)- X_W_avg(:,t));
            end
            SP_prev = SP_temp;
        end
        %X_pred(:,:,i) = 
    end
end