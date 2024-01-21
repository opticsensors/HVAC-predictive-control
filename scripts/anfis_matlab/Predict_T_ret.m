%% Time Series Data
train_x = dlmread('../../data/processed/train_x.csv',',');
train_y = dlmread('../../data/processed/train_y.csv',',');
test_x = dlmread('../../data/processed/test_x.csv',',');
test_y = dlmread('../../data/processed/test_y.csv',',');

train=cat(2,train_x,train_y);
test=cat(2,test_x,test_y);

s=size(test_y);
time = 1:s(1);
x = test_y(:, end);

figure(1)
plot(time,x)
title('Time Series Tret at t+1hour')
xlabel('Time (sec)')
ylabel('x(t)')


%% Build Initial Fuzzy System
fis = genfis(train_x,train_y,...
    genfisOptions('GridPartition'));

figure
subplot(2,2,1)
plotmf(fis,'input',1)
subplot(2,2,2)
plotmf(fis,'input',2)
subplot(2,2,3)
plotmf(fis,'input',3)
subplot(2,2,4)
plotmf(fis,'input',4)

%% Train ANFIS Model
options = anfisOptions('InitialFIS',fis,'ValidationData',test);
[fis1,error1,ss,fis2,error2] = anfis(train,options);
figure
subplot(2,2,1)
plotmf(fis2,'input',1)
subplot(2,2,2)
plotmf(fis2,'input',2)
subplot(2,2,3)
plotmf(fis2,'input',3)
subplot(2,2,4)
plotmf(fis2,'input',4)

%% Plot Errors Curves
figure
plot([error1 error2])
hold on
plot([error1 error2],'o')
legend('Training error','Checking error')
xlabel('Epochs')
ylabel('Root Mean Squared Error')
title('Error Curves')

%% Compare Original and Predicted Series
anfis_output = evalfis(fis2,[test_x]);

figure
index = 1:s(1);
plot(time(index),[x(index) anfis_output])
xlabel('Time (sec)')
title('MG Time Series and ANFIS Prediction')

%% Save resu

dataToSave = [time(index), x(index), anfis_output];
dlmwrite('../../data/raw/mgresults.txt', dataToSave, 'delimiter', ',', 'precision', '%10.6f');



