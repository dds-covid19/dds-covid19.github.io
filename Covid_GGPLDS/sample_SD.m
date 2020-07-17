function [D,S,S0] =sample_SD(X,W,Z,Phi_prime,S0,S,D,Lambda,Phi,Model_hyper_params,Model_properties)
    %F = W.*Z
    T = Model_properties.T; 
    K = Model_properties.K;
    X_dim_S = Model_properties.X_dim_S;
    N = Model_properties.N;
    K_prime = Model_properties.K_prime;
   S_0 = [S0 S(:,1:T-1)];

   
    F= W.*Z;
    temp_ch = chol((Model_hyper_params.Etta0 + F'*Phi_prime*F +1e-6*eye(K)));
    inv_temp = temp_ch\eye(K);
    z = mvnrnd(zeros(K,1),eye(K));
    S0 = inv_temp*z' + inv_temp*inv_temp'*(F'*Phi_prime*S(:,1) + Model_hyper_params.m0);
    for t= 1:T
        if t == T
            temp_ch = chol((Phi_prime + N(t)*D'*Phi*D));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(Phi_prime*F*S(:,t-1) + D'*Phi*sum(X{t},2) );
        elseif t>1
            temp_ch = chol((Phi_prime + N(t)*D'*Phi*D + F'*Phi_prime*F ));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(D'*Phi*sum(X{t},2) + F'*Phi_prime*S(:,t+1) + Phi_prime*F*S(:,t-1));        
        else
            temp_ch = chol((Phi_prime + N(t)*D'*Phi*D + F'*Phi_prime*F ));
            inv_temp = temp_ch\eye(K);
            z = mvnrnd(zeros(K,1),eye(K));
            S(:,t)= inv_temp*z' + inv_temp*inv_temp'*(D'*Phi*sum(X{t},2) + F'*Phi_prime*S(:,t+1) + Phi_prime*F*S0);
        end

    end
    
    F1 = zeros(K,1);
    for t= 1:T
       F1 = F1+N(t)*S(:,t).^2;
       Xts(:,t) =  sum(reshape((X{t}),X_dim_S,[]),2);
    end
    F2 = (S*Xts')';

    for k = 1:K
        F3(:,k) =  D(:,[1:k-1,k+1:K])*(S([1:k-1,k+1:K],:)*(N.*S(k,:))');
        temp_ch = chol(F1(k).*Phi+ Lambda(:,:,k));
        inv_temp = temp_ch\eye(X_dim_S);
        z = mvnrnd(zeros(X_dim_S,1),eye(X_dim_S));
        D(:,k)= inv_temp*z' + inv_temp*inv_temp'*(Phi*F2(:,k)-Phi*F3(:,k));

    end
end
