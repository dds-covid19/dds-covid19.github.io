
% SS is one step transformation after reordering 
sss=nchoosek(1:30,3);
ddd1= inf*ones(4060,1);
for iii =1:4060
    S_select = SS(sss(iii,:),:);
    D_select = D_new(:,sss(iii,:));
    ddd1(iii) = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_select*S_select(:,12:end)).^2,1)));
end
[vv1,loc1] = min(ddd1);
[fff1,ggg1] = sort(ddd1);
figure;plot3(SS(sss(loc1,1),:),SS(sss(loc1,2),:),SS(sss(loc1,3),:))


% find the one from original 
ddd2= inf*ones(4060,1);
for iii =1:4060
    S_select = S(sss(iii,:),:);
    D_select = D(:,sss(iii,:));
    ddd2(iii) = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_select*S_select(:,12:end)).^2,1)));
end
[vv2,loc2] = min(ddd2);
[fff2,ggg2] = sort(ddd2);
figure;plot3(S(sss(loc2,1),:),S(sss(loc2,2),:),S(sss(loc2,3),:))

% find the one from original but rotated
ddd3= inf*ones(4060,1);
for iii =1:4060
    S_select = S_new_r(sss(iii,:),:);
    D_select = D_new(:,sss(iii,:));
    ddd3(iii) = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_select*S_select(:,12:end)).^2,1)));
end
[vv3,loc3] = min(ddd3);
[fff3,ggg3] = sort(ddd3);
figure;plot3(S_new_r(sss(loc3,1),:),S_new_r(sss(loc3,2),:),S_new_r(sss(loc3,3),:))



%PCA before transition 
S_new_rzm = S_new_r - mean(S_new_r,2);
[S_p , U_p] = pca(S_new_rzm', 3);
Y1_p = U_p' * S_new_rzm;
S_new_recon = U_p *Y1_p;
figure;plot3(Y1_p(1,:),Y1_p(2,:),Y1_p(3,:))
e_bt_recon = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_new*S_new_recon(:,12:end)).^2,1)));



%PCA before transition 
S_new_rzm = S_new_r - mean(S_new_r,2);
[S_p , U_p] = pca(S_new_rzm', 3);
Y1_p = U_p' * S_new_rzm;
S_new_recon = U_p *Y1_p;
S_new_recon = S_new_recon +mean(S_new_r,2);
figure;plot3(Y1_p(1,:),Y1_p(2,:),Y1_p(3,:))
e_bt_recon = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_new*S_new_recon(:,12:end)).^2,1)));


%PCA after transition 
SS_rzm = SS - mean(SS,2);
[SS_p , US_p] = pca(SS_rzm', 3);
Y2_p = US_p' * SS_rzm;
SS_recon = US_p * Y2_p;
SS_recon = SS_recon +mean(SS,2);
figure;plot3(Y2_p(1,:),Y2_p(2,:),Y2_p(3,:))
e_at_recon = mean(sqrt(sum((X_orig_avg1(:,12:end)-D_new*SS_recon(:,12:end)).^2,1)));


figure(1);
figure(2);
for i =1:530
    for j = 1:30
    dd(j,i)= sqrt(sum((X_orig_avg1(:,i)-D_new(:,j)*S_new_r(j,i)).^2,1));
    end
    %[val(i) ind(i)] = min(dd(:,i));
    [val(i) ind(i)] = max(abs(S_new_r(:,i)));
    if ind(i)>=1 && ind(i)<=7
        com(i) = 1;
    elseif ind(i)>=8 && ind(i)<=10
        com(i) = 2;
    elseif ind(i)>=11 && ind(i)<=14
        com(i) =3;
    elseif ind(i)>=15 && ind(i)<=16
        com(i) =4;
    else
        com(i)=5;
    end
    if i<25
        figure(1);imagesc(abs(S_new_r(:,1:i)))
        figure(2);plot3(X(1:i),Y(1:i),Z(1:i))
    else
        figure(1);imagesc(abs(S_new_r(:,i-24:i)))
        figure(2);plot3(X(1:i),Y(1:i),Z(1:i))
    end
    pause
    i
    com(i)
    val(i)
    ind(i)
end

for j = 1:30
    dd(j,i)= sqrt(sum((X_orig_avg1(:,i)-D_new(:,j)*S_new_r(j,i)).^2,1));
end
[val(i) ind(i)] = min(d(:,i));
if ind(i)>=1 && ind(i)<=7
    com(i) = 1;
elseif ind(i)>=8 && ind(i)<=10
    com(i) = 2;
elseif ind(i)>=11 && ind(i)<=14
    com(i) =3;
elseif ind(i)>=15 && ind(i)<=16
    com(i) =4;
else
    com(i)=5;
end

figure()
JJ=3;
idx_11 = kmeans(S_new_r',JJ,'distance','correlation');
X_p = NaN(532,1);
Y_p = NaN(532,1);
Z_p = NaN(532,1);
total_count =0;
total_count1 =0;
% S_new_1(:,idx_11==1)=S_new_r(:,idx_11==1);
% S_new_2(:,idx_11==2)=S_new_r(:,idx_11==2);
% S_new_3(:,idx_11==3)=S_new_r(:,idx_11==3);
% S_new_4(:,idx_11==4)=S_new_r(:,idx_11==4);
S_new_i=zeros(30,533,JJ);
 for i =1:JJ
     ppdx = (idx_11==i);
     S_new_i(:,idx_11==i,i)= S_new_r(:,idx_11==i);
     X_p = NaN(533,1);
     Y_p = NaN(533,1);
     Z_p = NaN(533,1);
     X_p(ppdx) = X(ppdx);
     Y_p(ppdx) = Y(ppdx);
     Z_p(ppdx) = Z(ppdx);
     total_count = total_count+ sum(idx_11==1);
     total_count1 =total_count1+ sum(~isnan(X_p))
        for cc = 1:2
            if cc ==1
                subplot(JJ,2,2*(i-1)+cc)
                plot3(X_p,Y_p,Z_p)
                xlim([-25 25])
                ylim([-30 30])
                zlim([-5 55])
                xlabel('x_1','FontSize', 12) 
                ylabel('x_2','FontSize', 12) 
                zlabel('x_3','FontSize', 12)
                grid on
            elseif cc==2
                subplot(JJ,2,2*(i-1)+cc)
                imagesc(abs(S_new_i(:,:,i)))
                ylabel('x_t','FontSize', 12,'FontWeight','bold') 
                xlabel('t','FontSize', 12) 
            end
%             if cc==1
%                 title(['Cluster ',num2str(i)],'FontSize', 9)
%             end
%             if cc==1
%                 ylabel(['d',num2str(i)],'FontSize', 9)
%                 hYLabel = get(gca,'YLabel');
%                 set(hYLabel,'rotation',0,'VerticalAlignment','middle')
%                % title(['d',num2str(i)],'FontSize', 9,'position',[-200 0])
%             end

        end
       
    end

suptitle('Clustered States equivalency of latent space variables')
fig1=figure;
fig1.Renderer='Painters';

fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

   


figure(1);imagesc(abs(S_new_1))
figure(2);imagesc(abs(S_new_2))
figure(2);imagesc(abs(S_new_2))
figure(3);imagesc(abs(S_new_3))
figure(4);imagesc(abs(S_new_4))
figure(5);plot3(X(ppdx),Y(ppdx),Z(ppdx))
ppdx=(idx_11==2);
figure(5);plot3(X(ppdx),Y(ppdx),Z(ppdx))
figure(6);plot3(X(ppdx),Y(ppdx),Z(ppdx))
ppdx=(idx_1==1)
Undefined function or variable 'idx_1'.
 
Did you mean:
ppdx=(idx_11==1);
figure(5);plot3(X(ppdx),Y(ppdx),Z(ppdx))
ppdx=(idx_11==3);
figure(7);plot3(X(ppdx),Y(ppdx),Z(ppdx))
ppdx=(idx_11==4);
figure(8);plot3(X(ppdx),Y(ppdx),Z(ppdx))