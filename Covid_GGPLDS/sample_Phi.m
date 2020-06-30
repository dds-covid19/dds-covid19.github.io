function Phi = sample_Phi(Model_hyper_params,X_dim_S,N,X,S,D,Model_properties)
% sample Phi
    T = Model_properties.T; 
    K = Model_properties.K;
    X_dim_S = Model_properties.X_dim_S;
    N = Model_properties.N;
    K_prime = Model_properties.K_prime;
    G1 = zeros(X_dim_S);
    G2 = zeros(X_dim_S);
    G3 = zeros(X_dim_S);
    G4 = zeros(X_dim_S);

%     for t = 1:T
%         for i=1:N(t)
%              GGG = GGG + (X{t}(:,i)-D*S(:,t))*(X{t}(:,i)-D*S(:,t))';
%         end
%     end
    for t=1:T
      G1 = G1 + X{t}*X{t}';
      G2 = G2 + sum(X{t},2)*(S(:,t)'*D');% X{t}*repmat((S(:,t)'*D'),[size(X{t},2),1]);
      G3 = G3 + N(t)*D*(S(:,t)*S(:,t)')*D';  
      G4 = G4 + (D*S(:,t))*sum(X{t},2)';
    end
    GG = G1-G2+G3-G4;
    Phi_inv = iwishrnd(triu(GG)+triu(GG,1)'+Model_hyper_params.V1,Model_hyper_params.df1+Model_properties.N_sum);
    Phi = Phi_inv\eye(X_dim_S);
end

