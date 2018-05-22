function MAUC  = calMAUC( test_target, predLabel, M)
% Function calMAUC is used to compute the MAUC.
 MAUC=0;
label_num = size(test_target,1);
labelMat = repmat(linspace(1,label_num,label_num),size(test_target,2),1);
real = test_target'.*labelMat;
real = sum(real,2);
pred=predLabel'.*labelMat;
pred=sum(pred,2);

for ii=1:label_num
    realdist(ii)=length(find(real(:)==ii));
    preddist(ii)=length(find(pred(:)==ii));
end
realdist1=zeros(max(realdist),label_num);
preddist1=zeros(max(preddist),label_num);
for ii=1:label_num
    realdist1(1:realdist(ii),ii)=find(real(:)==ii);
    preddist1(1:preddist(ii),ii)=find(pred(:)==ii);
end
para=2/(label_num*(label_num-1));
auc=zeros(label_num);
for i=1:label_num
    for k=1:label_num
        if(i~=k)
            for ii=1:preddist(i)
                for jj=1:preddist(k)
                    if(M(preddist1(ii,i),i)>M(preddist1(jj,k),i))
                        auc(i,k)=auc(i,k)+1;
                    end
                    if(M(preddist1(ii,i),i)==M(preddist1(jj,k),i))
                        auc(i,k)=auc(i,k)+0.5;
                    end
                    if(M(preddist1(ii,i),i)<M(preddist1(jj,k),i))
                        auc(i,k)=auc(i,k);
                    end
                end
            end
        end
        if(preddist(i)*preddist(k)~=0)
            auc(i,k)=auc(i,k)/(preddist(i)*preddist(k));
        end
    end
end
for i=1:label_num
    for m=1:label_num
        if(i<m)
            MAUC=MAUC+((auc(i,m)+auc(m,i))/2);
        end
    end
end
MAUC=para*MAUC;
end

