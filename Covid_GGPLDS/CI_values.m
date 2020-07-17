function B = CI_values(C)
    C = sort(C ,1,'descend');
    B =[C(floor(size(C,1)*2.5/100)+1,:);C(floor(size(C,1)*975/1000),:)];
 
end