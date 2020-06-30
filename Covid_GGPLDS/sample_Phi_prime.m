function Phi_prime = sample_Phi_prime(Model_hyper_params,T,S,W,Z,S0,Model_properties)
    T = Model_properties.T; 
    K = Model_properties.K;
    X_dim_S = Model_properties.X_dim_S;
    N = Model_properties.N;
    K_prime = Model_properties.K_prime;
    par1 =  T/2+Model_hyper_params.alpha1;
    par2 = 1./(0.5 .* sum((S-(W.*Z)*[S0,S(:,1:T-1)]).^2,2)+Model_hyper_params.beta1);
    for k =1:K
      temp = gamrnd(par1,par2(k));
      Phi_prime(k,k) = temp;
    end
end