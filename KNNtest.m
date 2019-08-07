load digits
X=learn.P';
Y=learn.T';
MdlEucl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1)
cvemodel = crossval(MdlEucl)
cvemdlloss = kfoldLoss(cvemodel)
labelEucl = predict(MdlEucl,test.P')
fold=test.T'
count=0;
for i=1:1797
   if(labelEucl(i)==fold(i))
       count=count+1;  
    end
end
   div=count/1797;
   disp(div);

MdlHam = fitcknn(X,Y,'NumNeighbors',3,'distance','hamming','Standardize',1)
cvhmodel = crossval(MdlHam)
cvhmdlloss = kfoldLoss(cvhmodel)
labelHam = predict(MdlHam,test.P')
fold=test.T'
count2=0;
for i=1:1797
   if(labelHam(i)==fold(i))
       count2=count2+1;  
    end
end
   div2=count2/1797;
   disp(div2);