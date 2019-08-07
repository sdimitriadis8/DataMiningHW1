load('BatsiDimitriadisData.mat');

indices=crossvalind('Kfold',test,10); %χωρίζει τα classes σε 10 τυχαια υποσυνολα
cp=classperf(test);
acc=0; %arxikopoihsh accuracy 
for i= 1:10
    val=(indices==i); %κρατάει για τεστ το πρώτο
    train=~val; %τα υπόλοιπα για train
    ctree= fitctree(learn(train,:),test(train,:)); %φτιάχνει δυαδικό δέντρο απόφασης με το Traning set
    %view(ctree,'mode','graph')
    cvmodel=crossval(ctree);
    Classnew = predict(ctree,mean(learn(val,:)))%κανει predict παίρνοντας το test
end