            
function [XP_temp,abser_death,abser_cases,Rec_data] = Predict_multisteps_con(Para_sample,Theta_sample,T,T_test,V,L,pred_count,delta,X_W_avg,task)


XP_temp    =   zeros(V,pred_count,T_test,size(Para_sample,2));
Rec_data   =   zeros(V,   T,   size(Para_sample,2));
abser_death     =   zeros(V/2,T_test,size(Para_sample,2));
abser_cases     =   zeros(V/2,T_test,size(Para_sample,2));


    for num_sample = 1:size(Para_sample,2)
        
        Theta_tmp = Theta_sample{1,num_sample};
        Phi_tmp   = Para_sample{1,num_sample}.Phi;
        Pi_tmp    = Para_sample{1,num_sample}.Pi;
        delta_tmp = delta{1,num_sample};
  
        Lambda   =  bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1}* Theta_tmp{1});
        Rec_data(:,:,num_sample) = poissrnd(Lambda);  
            
        for cc = 1: pred_count 
            Theta_pred = cell(L,1);           
            Theta_last = cell(L,1);  
               % obtain the latent state at last time step for the observed data
               for l=1:L
                 Theta_last{l} = Theta_tmp{l}(:,end);                                            
               end
                         
                for tstep =1:T_test

                    for l = L:-1:1
                        if l==L
                            Theta_pred{l} = randg(Pi_tmp{l}*Theta_last{l});
                        else
                            Theta_pred{l} = randg(Phi_tmp{l+1}*Theta_pred{l+1}+Pi_tmp{l}*Theta_last{l});
                        end
                    end
                    
                    XP_temp(:,cc,tstep,num_sample) = poissrnd(bsxfun(@times,delta_tmp{1}(end)', Phi_tmp{1} * Theta_pred{1}));%²ÉÑù
                    Theta_last = Theta_pred;


                end


        end 
  
        
        
            XP_avg_temp = squeeze(mean(XP_temp,2));
            
            if strcmp(task,'prediction_historic')
                
                for tstep =1:T_test
                    abser_death(:,tstep,num_sample) = mean(abs(XP_avg_temp(1:V/2,tstep,num_sample)- X_W_avg(1:V/2,tstep))./(X_W_avg(1:V/2,tstep)+1),2);
                    abser_cases(:,tstep,num_sample) = mean(abs(XP_avg_temp(1+V/2:end,tstep,num_sample)- X_W_avg(1+V/2:end,tstep))./(X_W_avg(1+V/2:end,tstep)+1),2);
                
                end
            else
                
                abser_death   =   [];
                abser_cases   =   [];
            end        
 
     
            
    end


    
        
end
