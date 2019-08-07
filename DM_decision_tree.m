load('BatsiDimitriadisData.mat');

indices=crossvalind('Kfold',test,10); %������� �� classes �� 10 ������ ���������
cp=classperf(test);
acc=0; %arxikopoihsh accuracy 
for i= 1:10
    val=(indices==i); %������� ��� ���� �� �����
    train=~val; %�� �������� ��� train
    ctree= fitctree(learn(train,:),test(train,:)); %�������� ������� ������ �������� �� �� Traning set
    %view(ctree,'mode','graph')
    cvmodel=crossval(ctree);
    Classnew = predict(ctree,mean(learn(val,:)))%����� predict ���������� �� test
end