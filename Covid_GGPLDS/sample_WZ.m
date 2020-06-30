function [W,Z] = sample_WZ(S0,S,Z,W,Phi_prime,sigma,theta,r,psi,Model_hyper_params,Model_properties)
    %F = W.*Z;
   T = Model_properties.T;
   K = Model_properties.K;
   N = Model_properties.N;
   K_prime = Model_properties.K_prime;
   S_0 = [S0 S(:,1:T-1)];
   % WZS = F * S_0; 
   Z_prev = Z;
   W_temp = normrnd(0,sqrt(1./sigma));
   W(Z_prev==0) = W_temp(Z_prev==0);
   Tj= diag(S_0 * S_0');
   [indexi , indexj]= find(Z_prev>0);
   for ii = 1:length(indexi)
       i = indexi(ii);
       j = indexj(ii);
      
       S_ijt = S(i,:)-(W(i,:).*Z(i,:))*S_0+ (W(i,j).*Z(i,j).*S_0(j,:));
       Q(i,j) = S_ijt*S_0(j,:)';

       tau(i,j) = 1./(Z(i,j)*Phi_prime(i,i)*Tj(j)+sigma(i,j)); %change 5 1./
       if (tau(i,j) == inf )
           %tau(i,j) = 1;
           flag_tau = 1;
       end

        mu(i,j) = tau(i,j)*(Z(i,j)*Phi_prime(i,i)*Q(i,j));
        W(i,j) = normrnd(mu(i,j),sqrt(tau(i,j)));
   end
   
   for i = 1:K
        for j = 1:K
            S_ijt = S(i,:)-(W(i,:).*Z(i,:))*S_0+ (W(i,j).*Z(i,j).*S_0(j,:));
            Q(i,j) = S_ijt*S_0(j,:)';

            if (i == j)
                ln_p0(i,j) = -(theta(i,:)*diag(r)*psi(:,j));
                ln_p1(i,j) = (-0.5*Phi_prime(i,i)*(-2*W(i,j)*Q(i,j) + (W(i,j)^2)* Tj(j))) + log(1-exp(ln_p0(i,j)));%log(-ln_p0(i,j))+ln_p0(i,j)/2 -ln_p0(i,j)^2/24
                p_ratio(i,j) = exp(ln_p0(i,j)-ln_p1(i,j));
            else 
                ln_p0(i,j) = -(theta(i,:)*diag(r)*psi(:,j));
                ln_p1(i,j) = (-0.5*Phi_prime(i,i)*(-2*W(i,j)*Q(i,j) + (W(i,j)^2)* Tj(j))) + log(1-exp(ln_p0(i,j)));%log(-ln_p0(i,j))+ln_p0(i,j)/2 -ln_p0(i,j)^2/24
                p_ratio(i,j) = exp(ln_p0(i,j)-ln_p1(i,j));
            end
            Z(i,j) = binornd(1,1./(1+p_ratio(i,j))); %change 6 1./
%             if (isnan(Z(i,j)) || isinf(Z(i,j)))
%                 dddd=1;
%             end
            
            
        end
   end
end


               