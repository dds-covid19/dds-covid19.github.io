function [D,S,S0,l4,omega,rr] = sample_SD_Omega_rr_new(y, W, Z, Phi_prime, S0, S, D, omega, rr, l4, Model_hyper_params, Model_properties)
    %y= data;
    T = Model_properties.T; 
    K = Model_properties.K;
    X_dim_S = Model_properties.X_dim_S;
    N = Model_properties.N;
    K_prime = Model_properties.K_prime;
   S_0 = [S0 S(:,1:T-1)];
   Truncation = 5;
   alpha_r = Model_hyper_params.alpha_r;
   beta_r = Model_hyper_params.beta_r;
   
    
    
   %%%%%%%polyagamma sampling
   for p=1:X_dim_S
       for t=1:T
           omega(p,t)= PolyaGamRnd_Gam(y(p,t)+rr,D(p,:)*S(:,t),Truncation);
       end
   end
   %%%%%%%%%%% define vector v
   V = (y-rr)/2.0;
   
   %%%%%%%sample St
    F= W.*Z;
    temp_ch = chol((Model_hyper_params.Etta0 + F'*Phi_prime*F +1e-6*eye(K)));
    inv_temp = temp_ch\eye(K);
    z = mvnrnd(zeros(K,1),eye(K));
    S0 = inv_temp*z' + inv_temp*inv_temp'*(F'*Phi_prime*S(:,1) + Model_hyper_params.m0);
    for t= 1:T
        if t == T
            temp_ch = chol((Phi_prime + N(t)*D'*diag(omega(:,t))*D));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(Phi_prime*F*S(:,t-1) + D'*V(:,t) );
        elseif t>1
            temp_ch = chol((Phi_prime + N(t)*D'*diag(omega(:,t))*D + F'*Phi_prime*F ));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(D'* V(:,t) + F'*Phi_prime*S(:,t+1) + Phi_prime*F*S(:,t-1));        
        else
            temp_ch = chol((Phi_prime + N(t)*D'*diag(omega(:,t))*D + F'*Phi_prime*F ));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(D'* V(:,t) + F'*Phi_prime*S(:,t+1) + Phi_prime*F*S0);
        end

    end
    

    
    %%%%%sample dp
    for p=1:X_dim_S
        temp_ch = chol(S*diag(omega(p,:))*S'+ sqrt(K)*eye(K));%sqrt(X_dim_S)*eye(X_dim_S));
        inv_temp = temp_ch\eye(K);
        z = mvnrnd(zeros(K,1),eye(K));
        D(p,:)= (inv_temp*z' + inv_temp*inv_temp'*(S*V(p,:)'))';
    end
    
        %%%%% sample dk
%     F1 = omega * (S.^2)' ;
%     for k =1:K
%         F2(:,k)= V*S(k,:)';
%         Psi_k =  D(:,[1:k-1,k+1:K])*S([1:k-1,k+1:K],:);
%         F3(:,k) = diag ((omega.*S(k,:))*Psi_k');
%         temp_ch = chol(diag(F1(:,k))+ sqrt(X_dim_S)*eye(X_dim_S));
%         inv_temp = temp_ch\eye(X_dim_S);
%         z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S));
%         D(:,k)= inv_temp*z' + inv_temp*inv_temp'*(F2(:,k)-F3(:,k));
%         
%     end
    
     %%%%sample crt
    %sample l4
    for p = 1:X_dim_S
        par2 =rr;
        for t = 1:T 
           par1 = y(p,t);
           l4(p,t) = CRT_sum_mex(par1,par2);
        end
    end
    Psi = D * S;
    %%%%% samplr rr
    l4_ss = sum(sum(l4,2));
    par1 = l4_ss + alpha_r;
    p4 = sum( sum(log(1+exp(Psi)),2));
    par2 =1/(beta_r+p4);
    rr = gamrnd(par1,par2)
    
    
    
    
    %%%%%% sample dk
%     F1 = omega * (S.^2)' ;
%     for k =1:K
%         F2(:,k)= V*S(k,:)';
%         Psi_k =  D(:,[1:k-1,k+1:K])*S([1:k-1,k+1:K],:);
%         F3(:,k) = diag ((omega.*S(k,:))*Psi_k');
%         temp_ch = chol(diag(F1(:,k))+ Lambda(:,:,k));%sqrt(X_dim_S)*eye(X_dim_S));
%         inv_temp = temp_ch\eye(X_dim_S);
%         z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S));
%         D(:,k)= inv_temp*z' + inv_temp*inv_temp'*(F2(:,k)-F3(:,k));
%         
%     end
    
   
  
end
