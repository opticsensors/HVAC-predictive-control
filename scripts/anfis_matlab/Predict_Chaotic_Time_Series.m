%% Time Series Data
filename = '../../data/raw/mgdata.txt';
mgdata = dlmread(filename,',');
time = mgdata(:,1);
x = mgdata(:, 2);

figure(1)
plot(time,x)
title('Mackey-Glass Chaotic Time Series')
xlabel('Time (sec)')
ylabel('x(t)')

%% Preprocess Data
for t = 118:1117 
    Data(t-117,:) = [x(t-18) x(t-12) x(t-6) x(t) x(t+6)]; 
end
trnData = Data(1:500,:);
chkData = Data(501:end,:);


%% Build Initial Fuzzy System
fis = genfis(trnData(:,1:end-1),trnData(:,end),...
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
options = anfisOptions('InitialFIS',fis,'ValidationData',chkData);
[fis1,error1,ss,fis2,error2] = anfis(trnData,options);
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
anfis_output = evalfis(fis2,[trnData(:,1:4); chkData(:,1:4)]);

figure
index = 124:1123;
plot(time(index),[x(index) anfis_output])
xlabel('Time (sec)')
title('MG Time Series and ANFIS Prediction')

%% Calculate and plot the prediction error.
diff = x(index) - anfis_output;
plot(time(index),diff)
xlabel('Time (sec)')
title('Prediction Errors')

%% 6 future samples
index1=1118:1123;
figure
plot(time(index1+1), [x(index1) anfis_output(end-5:end)])
hold on;
plot(time(index1+1), [x(index1) anfis_output(end-5:end)], 'o', 'MarkerEdgeColor', 'k')
xticks(index1)
xlabel('Time')
title('Next 6 samples prediction and actual MG series')

%% Save resu

dataToSave = [time(index), x(index), anfis_output];
dlmwrite('../../data/raw/mgresults.txt', dataToSave, 'delimiter', ',', 'precision', '%10.6f');



