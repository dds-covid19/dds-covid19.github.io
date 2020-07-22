function X_recon_avg1 = data_reconstruction(X,X_w,S0_col,S_col,Phi_prime_col,W_col,Z_col,Phi_col,D_col)
    
    
    for i = 1:size(X,2)
       TRdata(:,i)= X{i}(:,1); 
    end
    for i = 1:size(X_w,2)
       TEdata(:,i)= mean(X_w{i},2);
    end
    initV = cov(S0_col');
    for i = 1:size(S_col,3)
        for t = 1:size(X,2) 

            if t == 1
                SP_temp = (W_col(:,:,i).*Z_col(:,:,i))*S0_col(:,i); 
                XP_temp = (D_col(:,:,i)*SP_temp);
                XP4(:,t,i) = mean(XP_temp,2);
            else
                [S, V, VV, loglik] = kalman_filter(TRdata(:,1:t-1), W_col(:,:,i).*Z_col(:,:,i), D_col(:,:,i), inv(Phi_prime_col(:,:,i)), inv(Phi_col(:,:,i)), S0_col(:,i), initV);
                SP_temp = (W_col(:,:,i).*Z_col(:,:,i))*S(:,end);
                XP_temp = (D_col(:,:,i)*SP_temp);
                %XP_temp = mvnrnd((D_col(:,:,i)*SP_temp)',inv(Phi_col(:,:,i)),pred_count)';
                XP4(:,t,i) = mean(XP_temp,2);
            end
            SP_prev = SP_temp;
        end
        %X_pred(:,:,i) = 
    end
    X_recon_avg1 = mean(XP4,3);

end