            
function [XP_temp,abser,Rec_data] = Predict_multisteps_singlelayer(Para_sample,Theta_sample,T,T_test,V,L,pred_count,delta,X_W_avg,task)


XP_temp    =   zeros(V,pred_count,T_test,size(Para_sample,2));
Rec_data   =   zeros(V,   T,   size(Para_sample,2));
abser     =   zeros(V,T_test,size(Para_sample,2));


     for num_sample = 1:size(Para_sample,2)
        
        Theta_tmp = Theta_sample{1,num_sample};
        Phi_tmp   = Para_sample{1,num_sample}.Phi;
        Pi_tmp    = Para_sample{1,num_sample}.Pi;
        delta_tmp = delta{1,num_sample};
  
        Lambda   =  bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1}* Theta_tmp{1});
        Rec_data(:,:,num_sample) = poissrnd(Lambda);  
            
                for tstep =1:T_test
                        % obtain the latent state at last time step for the observed data
                             
                    if tstep == 1
                       for cc = 1:pred_count
                           Theta_pred(:,cc) = (Pi_tmp{1}*Theta_tmp{1}(:,end));
                           XP_temp(:,cc,tstep,num_sample) = (bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1} *Theta_pred(:,cc)));%采样
                       end

                    else
                        
                        for cc = 1:pred_count
                           Theta_pred(:,cc) = (Pi_tmp{1}*Theta_last(:,cc));
                           XP_temp(:,cc,tstep,num_sample) = (bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1} *Theta_pred(:,cc)));%采样
                        end  
                       
                    end
                            
                      Theta_last = Theta_pred;

                   
                      XP_avg_temp = squeeze(mean(XP_temp,2));
    
                       if strcmp(task,'prediction_historic')                
                          abser(:,tstep,num_sample) = mean(abs(XP_avg_temp(:,tstep,num_sample)- X_W_avg(:,tstep))./(X_W_avg(:,tstep)+1),2);                            
                       else
                          abser=[];
                       end  

                end


         
  
        
        
            
      
 
     
            
    end

     



%  A = zeros(15,7);  
%  C = zeros(51,7);  
%  B=Theta_tmp{1}(:,end);
%  for t =1:7
%     A(:,t) = randg(Pi_tmp{1}*B);
%     C(:,t) = poissrnd(bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1} *Theta_pred(:,cc)));
%     B = squeeze(A(:,t));
%  end
%   tmp =sum(Lambda,1);
%   D= cat(2,tmp, sum(C,1)  )   ;
end
