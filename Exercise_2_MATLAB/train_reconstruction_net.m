% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all


%% Load training data
% load file "DATA_MMF_28.mat"

load("DATA_MMF_16_aug.mat")
%load("DATA_MMF_16.mat")

%% Create Neural Network Layergraph MLP

inputDim = size(XTrain); %Dimension des Input
outputDim = size(YTrain); %Dimensionen Output
I_px = inputDim(1);
O_px = outputDim(1);

layers = [
    imageInputLayer([I_px I_px 1],'Name','Input')
fullyConnectedLayer(I_px^2,'Name','Fc1')
reluLayer('Name','Relu1')
fullyConnectedLayer(O_px^2,'Name','Fc2')
reluLayer('Name','Relu2')
depthToSpace2dLayer([O_px O_px],'Name','dts1')
regressionLayer('Name','Output')
];

%analyzeNetwork(layers)
%% Training network
% define "trainingOptions"

options = trainingOptions("adam");

options.MiniBatchSize = 128;

options.MaxEpochs = 50;

options.InitialLearnRate = 0.001;

options.ExecutionEnvironment = 'auto';

options.OutputNetwork = 'best-validation-loss';

options.ValidationData = {XValid, YValid};

options.Plots = 'training-progress';

options.ValidationPatience = 70;

% training using "trainNetwork"

mlp = trainNetwork(XTrain,YTrain,layers,options);

%% Calculate Prediction 
% use command "predict"
Ypred = predict(mlp,XTest);

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR




% Anzeigen einiger Ergebnisse für visuelle Kontrolle 
% k=0;
% for i=1:10
%     k = k+1;
%     subplot(10,3,k), imshow(XTest(:,:,:,i),[0 255]),title('Input')
%     k = k+1;
%     subplot(10,3,k), imshow(YTest(:,:,:,i),[0 255]),title('Output')
%     k = k+1;
%     subplot(10,3,k), imshow(Ypred(:,:,:,i),[0 255]),title('Output Prediction')
% end


ypredDim = size(Ypred);

% RSME pro Bild und Durchschnitt
Pred_rmse = rmse(Ypred(),single(YTest()),[1 2]);
tmp = 0;
for i=1:size(Pred_rmse,4)
    tmp = tmp + Pred_rmse(:,:,1,i);
end
Pred_rmse_d = tmp/size(Pred_rmse,4);

for i = 1 : ypredDim(4)
     
    Pred_ssim(i) = ssim(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_psnr(i) = psnr(Ypred(:,:,1, i),single(YTest(:,:,1,i)));    
    Pred_corr(i) = corr2(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
 
end

%Durchsnitt SSIM, PSNR, CORR
tmp=0;
for i=1:size(Pred_ssim,2)
    tmp = tmp + Pred_ssim(1,i);
end
Pred_ssim_d = tmp/size(Pred_ssim,2);

tmp=0;
for i=1:size(Pred_psnr,2)
    tmp = tmp + Pred_psnr(1,i);
end
Pred_psnr_d = tmp/size(Pred_psnr,2);

tmp=0;
for i=1:size(Pred_corr,2)
    tmp = tmp + Pred_corr(1,i);
end
Pred_corr_d = tmp/size(Pred_corr,2);


%% Boxplots for step 6 of instructions
boxchart(Pred_rmse)

%% Step 7: create Neural Network Layergraph U-Net

layers = unetLayers([I_px I_px 1],2,'encoderDepth',3);
 
finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name','Final-ConvolutionLayer');
layers = replaceLayer(layers,'Final-ConvolutionLayer',finalConvLayer);

layers = removeLayers(layers,'Softmax-Layer');

regLayer = regressionLayer('Name','Reg-Layer');
layers = replaceLayer(layers,'Segmentation-Layer',regLayer);

layers = connectLayers(layers,'Final-ConvolutionLayer','Reg-Layer');

%analyzeNetwork(layers)

unet = trainNetwork(XTrain,YTrain,layers,options);

%% Evaluate Network
Ypred = predict(unet,XTest);
k=0;
for i=1:10
    k = k+1;
    subplot(10,3,k), imshow(XTest(:,:,:,i),[0 255]),title('Input')
    k = k+1;
    subplot(10,3,k), imshow(YTest(:,:,:,i),[0 255]),title('Output')
    k = k+1;
    subplot(10,3,k), imshow(Ypred(:,:,:,i),[0 255]),title('Output Prediction')
end

%% Boxplots for step 8 of instructions
