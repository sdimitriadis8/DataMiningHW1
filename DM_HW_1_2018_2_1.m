%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%&%                                                   %&%
%&%             Λ.07 - ΕΞΟΡΥΞΗ ΔΕΔΟΜΕΝΩΝ              %&%
%&%                1Η Σειρά Ασκήσεων                  %&%
%&%                                                   %&%
%&%                   Υλοποίηση:                      %&%
%&%                                                   %&%
%&%             Μπάτση Σοφία     Α.Μ.:372             %&%
%&%         Δημητριάδης Σωκράτης Α.Μ.:359             %&%
%&%                                                   %&%
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%%
clear all;
clc;

% Φόρτωση του αρχείου δεδομένων/καταχωρήσεων
load('BatsiDimitriadisData2.mat');

% Υπολογισμός των διαστάσεων του πίνακα δεδομένων X
[m,n] = size(X) ;

% Αρχικοποίηση μεταβλητών-πινάκων που θα χρειαστούν
A = zeros(m,n);
B = zeros(100,14,10);

% Για να πραγματοποιήσουμε 10-fold cross-validation
% θα χρειαστούμε 10 τυχαίως παραγόμενα και
% ξένα-μεταξύ-τους υποσύνολα του συνόλου δεδομένων.
% Κάθε φορά θα χρησιμοποιούνται τα 9 (k-1)
% υποσύνολα ως σύνολο εκπαίδευσης, ενώ το
% εναπομείναν υποσύνολο ως σύνολο ελέγχου του
% μοντέλου. Για να γίνουν τα παραπάνω θα χρειαστεί
% μια προεργασία:

% 1. Κατασκευή ενός διανύσματος τυχαίων μεταθέσεων, για να
% γίνει χρήση των τιμών του ως (τυχαίους) δείκτες
index = randperm(m);

% 2. Τυχαία μετάθεση των γραμμών του πίνακα X των δεδομένων
% για να κατασκευαστούν στη συνέχεια τα 10 ξένα υποσύνολά
% του, κατά τυχαίο τρόπο
for i=1:m

    A(i,:)=X(index(i),:);

end

% 3. Κατασκευή των 10 ξένων υποσυνόλων κληρονομώντας
% την τυχαιότητα από την κατασκευή του πίνακα A
for i=1:10
    for j=1:100

        B(j,:,i)=A(j,:);

    end
end

% Αρχικοποίηση της απώλειας ενός 
% ταξινομητή στο 100%
classloss = 1;

knnresults = zeros(20,5);
SVMresults = zeros();

bestloss = zeros(20,1);
bestneighbors = zeros(20,1);

% %% 
% 
% % k-NN
% 
% % Επανάληψη της διαδικασίας για εύρεση
% % (κατά προσέγγιση) του πλήθους γειτόνων 
% % με την μεγαλύτερη απόδοση 
% for j=1:20
%     
%     % Επαναληπτική εκπαίδευση του
%     % μοντέλου ταξινόμησης με 
%     % επιλογή περιττού πλήθους
%     % γειτόνων από έναν έως και εννέα
%     for i=1:2:9
%         
%             % Εκπαιδεύονται 5 ταξινομητές...
%             KnnMdl = fitcknn(X,Y,'NumNeighbors',i,'Standardize',1);
%             %... και επιλέγεται ο αποδοτικότερος 
%             % με 10-fold cross-validation...
%             cvknnMdl = crossval(KnnMdl);
%             cvknnloss = kfoldLoss(cvknnMdl);
%             knnresults(j,(i+1)/2) = cvknnloss;
%             % ... και ελέγχοντας κάθε φορά
%             % το ποσοστό της απώλειας
%             if cvknnloss < classloss
%                 classloss = cvknnloss;
%                 neighbors = i;
%             end   
%             
%     end
% 
%     % Διατηρούνται η απώλεια καθώς και
%     % το πλήθος γειτόνων του αποδοτικότερου
%     % ταξινομητή
%     bestneighbors(j) = neighbors;
%     bestloss(j) = bestloss(j)+classloss;
%     
%     % Επαναφορά της απώλειας στο 100%
%     % για εκ νέου συγκρίσεις
%     classloss = 1;
% end
% 
% % Υπολογίζεται ο μέσος όρος των γειτόνων 
% % και των απωλειών των αποδοτικότερων
% % εξ αυτών που εκπαιδεύτηκαν προηγουμένως
% meanneighbors = sum(bestneighbors)/20;
% meanloss = sum(bestloss)/20;
% disp(meanneighbors);
% disp(meanloss);
% %% 
% 
% % Naive Bayes
% 
% % Eκπαίδευση του ταξινομητή N-B
% % υποθέτοντας κανονική κατανομή
% NBMdl = fitcnb(X,Y,'Distribution','normal');
% cvNBMdl = crossval(NBMdl);
% cvNBLoss = kfoldLoss(cvNBMdl);
% 
% disp(cvNBLoss);
%%

% Support Vector Machines

% Eκπαίδευση του SVM ταξινομητή με RBF
% συνάρτηση πυρήνα: 

% Επαναληπτική εκπαίδευση του
% μοντέλου ταξινόμησης με 
% διαφορετική επιλογή πλάτους
sigma = zeros(1,6);

for i=-2:2:8
        
        gamma = 10^(-i);
        SVMRBFMdl = fitcsvm(X,Y,'KernelFunction','RBF','KernelScale',1/sqrt(gamma));
        cvSVMRBFMdl = crossval(SVMRBFMdl);
        cvSVMRBFLoss = kfoldLoss(cvSVMRBFMdl);
        
        SVMresults((i+4)/2) = cvSVMRBFLoss;
        sigma((i+4)/2) = 1/sqrt(gamma);
        
        % Κάθε φορά επιλέγεται ο ταξινομητής με
        % τη μικρότερη απώλεια.
        % Επίσης, διατηρούνται η απώλεια καθώς και
        % το σ του αποδοτικότερου ταξινομητή
        if cvSVMRBFLoss < classloss
            classloss = cvSVMRBFLoss;
            bestsigma = 1/sqrt(gamma);
        end
            
end

disp(classloss);

% Eκπαίδευση του αντίστοιχου SVM 
% ταξινομητή με γραμμικό πυρήνα (default):
SVMLinMdl = fitcsvm(X,Y);
cvSVMLinMdl = crossval(SVMLinMdl);
cvSVMLinLoss = kfoldLoss(cvSVMLinMdl);

disp(cvSVMLinLoss);

% %%
% 
% % Δέντρα απόφασης
% 
% % Eκπαίδευση του δέντρου απόφασης
% DTMdl = fitctree(X,Y);
% cvDTMdl = crossval(DTMdl);
% cvDTLoss = kfoldLoss(cvDTMdl);
% 
% disp(cvDTLoss);
% 
% view(DTMdl,'mode','graph')
