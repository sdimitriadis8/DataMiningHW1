%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%&%                                                   %&%
%&%             �.07 - ������� ���������              %&%
%&%                1� ����� ��������                  %&%
%&%                                                   %&%
%&%                   ���������:                      %&%
%&%                                                   %&%
%&%             ������ �����     �.�.:372             %&%
%&%         ����������� �������� �.�.:359             %&%
%&%                                                   %&%
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%%
clear all;
clc;

% ������� ��� ������� ���������/������������
X = load('BatsiDimitriadis_ClassData.txt');


% ����������� ��� ���������� ��� ������ ��������� X
[m,n] = size(X) ;

Y = X(:,5);
X = X(:,1:4);

% ������������ ��� �������� ���� 
% ���������� ��� 100%
classloss = 1;

knnresults = zeros(20,5);
SVMresults = zeros();

bestloss = zeros(20,1);
bestneighbors = zeros(20,1);

%% 

% k-NN

% ��������� ��� ����������� ��� ������
% (���� ����������) ��� ������� �������� 
% �� ��� ���������� ������� 
for j=1:20
    
    % ������������ ���������� ���
    % �������� ����������� �� 
    % ������� �������� �������
    % �������� ��� ���� ��� ��� �����
    for i=1:2:9
        
            % ������������� 5 �����������...
            KnnMdl = fitcknn(X,Y,'NumNeighbors',i,'Standardize',1);
            %... ��� ���������� � �������������� 
            % �� 10-fold cross-validation...
            cvknnMdl = crossval(KnnMdl);
            cvknnloss = kfoldLoss(cvknnMdl);
            knnresults(j,(i+1)/2) = cvknnloss;
            % ... ��� ���������� ���� ����
            % �� ������� ��� ��������
            if cvknnloss < classloss
                classloss = cvknnloss;
                neighbors = i;
            end
            
    end

    % ������������ � ������� ����� ���
    % �� ������ �������� ��� ��������������
    % ����������
    bestneighbors(j) = neighbors;
    bestloss(j) = bestloss(j)+classloss;
    
    % ��������� ��� �������� ��� 100%
    % ��� �� ���� ����������
    classloss = 1;
end

% ������������ � ����� ���� ��� �������� 
% ��� ��� �������� ��� ��������������
% �� ����� ��� ������������� ������������
meanneighbors = sum(bestneighbors)/20;
meanloss = sum(bestloss)/20;
disp(meanneighbors);
disp(meanloss);
%% 

% Naive Bayes

% E��������� ��� ���������� N-B
% ����������� �������� ��������
NBMdl = fitcnb(X,Y,'Distribution','normal');
cvNBMdl = crossval(NBMdl);
cvNBLoss = kfoldLoss(cvNBMdl);

disp(cvNBLoss);
%%

% Support Vector Machines

% E��������� ��� SVM ���������� �� RBF
% ��������� ������: 

% ������������ ���������� ���
% �������� ����������� �� 
% ����������� ������� �������
sigma = zeros(1,6);

for i=-2:2:8
        
        gamma = 10^(-i);
        SVMRBFMdl = fitcsvm(X,Y,'KernelFunction','RBF');
        cvSVMRBFMdl = crossval(SVMRBFMdl);
        cvSVMRBFLoss = kfoldLoss(cvSVMRBFMdl);
        
        SVMresults((i+4)/2) = cvSVMRBFLoss;
        sigma((i+4)/2) = 1/sqrt(gamma);
        
        % ���� ���� ���������� � ����������� ��
        % �� ��������� �������.
        % ������, ������������ � ������� ����� ���
        % �� � ��� �������������� ����������
        if cvSVMRBFLoss < classloss
            classloss = cvSVMRBFLoss;
            bestsigma = 1/sqrt(gamma);
        end
            
end

disp(classloss);

% E��������� ��� ����������� SVM 
% ���������� �� �������� ������ (default):
SVMLinMdl = fitcsvm(X,Y);
cvSVMLinMdl = crossval(SVMLinMdl);
cvSVMLinLoss = kfoldLoss(cvSVMLinMdl);

disp(cvSVMLinLoss);

%%

% ������ ��������

% E��������� ��� ������� ��������
DTMdl = fitctree(X,Y);
cvDTMdl = crossval(DTMdl);
cvDTLoss = kfoldLoss(cvDTMdl);

disp(cvDTLoss);

view(DTMdl,'mode','graph');
