function B = CI_values_all(C)
    C = sort(C ,1,'ascend');
    list =[0.01, 0.025, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99 ];
    B=[];
    for i =1:length(list)
       temp =  C(floor(size(C,1)*list(i))+1,:);
       B=[B;temp];
    end
    %B =[C(floor(size(C,1)*1.0/100)+1,:);C(floor(size(C,1)*2.5/100),:);C(floor(size(C,1)*5.0/100),:);C(floor(size(C,1)*10.0/100),:);C(floor(size(C,1)*15/100),:);C(floor(size(C,1)*20.0/100),:);C(floor(size(C,1)*25.0/100),:);C(floor(size(C,1)*30.0/100),:);C(floor(size(C,1)*35.0/100),:);C(floor(size(C,1)*40.0/100),:);C(floor(size(C,1)*45.0/100),:);C(floor(size(C,1)*50/100),:);C(floor(size(C,1)*55.0/100),:);C(floor(size(C,1)*60.0/100),:)];C(floor(size(C,1)*65.0/100),:)%;C(floor(size(C,1)*70.0/100),:);C(floor(size(C,1)*75.0/100),:);C(floor(size(C,1)*80.0/100),:);C(floor(size(C,1)*85.0/100),:);C(floor(size(C,1)*90.0/100),:);C(floor(size(C,1)*95.0/100),:);C(floor(size(C,1)*97.5/100),:);C(floor(size(C,1)*99.0/100),:)]
 
end