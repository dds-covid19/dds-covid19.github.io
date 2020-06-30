function m_ikj = sample_M_ikj(Z,theta,r,psi,t_pos,Model_properties)  
    K = Model_properties.K;
    K_prime = Model_properties.K_prime;
    lambda_ij = theta*diag(r)*psi;
    for i=1:K
        for j=1:K
            if (Z(i,j)== 0)
                m(i,j) = 0;
            else
                t_pos.lambda = lambda_ij(i,j);
            % to aviod truncated poisson generates Inf
                if t_pos.lambda< 1e-10
                   t_pos.lambda = 0;
                end
            % to avoid NAN being generated 
                if (t_pos.lambda >0)
                    m(i,j) =  random(t_pos,1,1);
                else
                    m(i,j) = 0;
                end
            end
        end
    end
      %m_ikj
    for i=1:K
        for j=1:K
            par1 = m(i,j);
            par2 =((r(:).*theta(i,:)').*psi(:,j))/(theta(i,:)*diag(r)*psi(:,j));
            if isnan(par2) 
                m_ikj(i,:,j) = zeros(1,K_prime);
            elseif par1 ==0 
                m_ikj(i,:,j) = zeros(1,K_prime);
            elseif all(par2 == 0)
                m_ikj(i,:,j) = zeros(1,K_prime);
            else 
                m_ikj(i,:,j) = mnrnd(par1,par2);
            end

        end
    end
end



