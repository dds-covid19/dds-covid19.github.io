function [theta,psi,r,l1,l2,l3,rho,tao,e,f,g] = sample_theta_psi_r_rho_tao_etta_e_f_g(m_ikj,theta,rho,e,r,f,g,l2,l3,psi,tao,Model_properties,Model_hyper_params)
    K = Model_properties.K;
    K_prime = Model_properties.K_prime;
    g0 = Model_hyper_params.g0;
    l0 = Model_hyper_params.l0;
    n0 = Model_hyper_params.n0;
    v0 = Model_hyper_params.v0;
    p0 = Model_hyper_params.p0;
    w0 = Model_hyper_params.w0;
    u0 = Model_hyper_params.u0;
    m00 = Model_hyper_params.m00;
    h0 = Model_hyper_params.h0;
    neu0 = Model_hyper_params.neu0;
    ksi0 = Model_hyper_params.ksi0; 
    gamma0 = Model_hyper_params.gamma0;

   

    % sample theta_ik"
    m_iks = sum(m_ikj,3);
    psi_ks = sum(psi,2);
    for i = 1:K
        for k_prime = 1:K_prime 
           par1 = m_iks(i,k_prime)+rho(i);
           par2 =1/(e(k_prime)+r(k_prime)*psi_ks(k_prime));
           theta(i,k_prime) = gamrnd(par1,par2);
        end
    end
    
     % sample psi_ik"
    m_skj = squeeze(sum(m_ikj,1));
    theta_sk = sum(theta,1);
    for j = 1:K
        for k_prime = 1:K_prime 
           par1 = m_skj(k_prime,j)+tao(j);
           par2 =1/(f(k_prime)+r(k_prime)*theta_sk(k_prime));%(f(k_prime)+r(k_prime)*theta_sk(k_prime));
           psi(k_prime,j) = gamrnd(par1,par2);
        end
    end
    
    % sample r_k
    m_sks = sum(sum(m_ikj,3),1);
    theta_sk = sum(theta,1);
    psi_ks = sum(psi,2);
    for k_prime = 1:K_prime 
       par1 = m_sks(k_prime)+neu0/K_prime;%neu0/K_prime;%etta(k_prime);
       par2 = 1/(g+psi_ks(k_prime)*theta_sk(k_prime));
       r(k_prime) = gamrnd(par1,par2);
    end
    


    %sample l1
    for i = 1:K
        par2 =rho(i);
        for k_prime = 1:K_prime 
           par1 = m_iks(i,k_prime);
           l1(i,k_prime) = CRT_sum_mex(par1,par2);
        end
    end

    % sample rho_i
    l1_is = sum(l1,2);
    p1 = sum(log(1-(r.* psi_ks) ./(e+r.* psi_ks))) ;
    par2 =1/(g0-p1);
    for i = 1:K 
       par1 = l1_is(i) + gamma0/(2*K);
       rho(i) = gamrnd(par1,par2);
    end

        %sample l2
    for j = 1:K
        par2 = tao(j);
        for k_prime = 1:K_prime 
           par1 = m_skj(k_prime,j);
           l2(k_prime,j) = CRT_sum_mex(par1,par2);
        end
    end

    % sample tao_j
    l2_sj = sum(l2,1);
    p2 = sum(log(1-(r.* theta_sk')./(f+r.* theta_sk'))) ;
    par2 =1/(h0-p2);
    for j = 1:K 
       par1 = l2_sj(j) + ksi0/K;
       tao(j) = gamrnd(par1,par2);
    end

    %sample l3
%     for k_prime = 1:K_prime
%        par1 = m_sks(k_prime);
%        par2 = etta(k_prime);
%        l3(k_prime) = CRT_sum_mex(par1,par2);
%     end

    % sample etta_k
%     for k_prime = 1:K_prime 
%         par1 = l3(k_prime) + neu0/K_prime;
%         par2 = 1/(l0-(log(1-psi_ks(k_prime)* theta_sk(k_prime) /(g+psi_ks(k_prime)* theta_sk(k_prime))))); 
%         etta(k_prime) = gamrnd(par1,par2);
%     end

    % sample e, f, g
    for k_prime = 1:K_prime
        par1 =  sum(rho)+m00;
        par2 = 1/(theta_sk(k_prime)+n0);
        e(k_prime) = gamrnd(par1,par2);
    end
    
    for k_prime = 1:K_prime
        par1 =  sum(tao)+u0;
        par2 = 1/(psi_ks(k_prime)+v0);
        f(k_prime) = gamrnd(par1,par2);
    end

%     par1 =  K_prime*sum(rho)+m00;
%     par2 = 1/(sum(theta_sk)+n0);
%     e = gamrnd(par1,par2);
%     
%     par1 =  K_prime*sum(tao)+u0;
%     par2 = 1/(sum(psi_ks)+v0);
%     f = gamrnd(par1,par2);

    par1 =  neu0+w0;
    par2 = 1/(sum(r)+p0);
    g = gamrnd(par1,par2);
end

