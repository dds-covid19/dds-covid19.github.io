function sigma = sample_sigma(Model_hyper_params,W,Model_properties)
    T = Model_properties.T; 
    K = Model_properties.K;
    X_dim_S = Model_properties.X_dim_S;
    N = Model_properties.N;
    K_prime = Model_properties.K_prime;
    par1 = Model_hyper_params.alpha0+0.5;
    par2 = 1./(Model_hyper_params.beta0 + W.^2/2);
    for i=1:K
        for j=1:K
            sigma(i,j) = gamrnd(par1,par2(i,j));
            %sigma2(i,j)= 1/temp;
        end
    end
end