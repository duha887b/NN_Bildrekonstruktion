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

options.MaxEpochs = 100;

options.InitialLearnRate = 0.001;

options.ExecutionEnvironment = 'auto';

options.OutputNetwork = 'best-validation-loss';

options.ValidationData = {XValid, YValid};

options.Plots = 'training-progress';

options.ValidationPatience = 100;

% training using "trainNetwork"

trainedNet = trainNetwork(XTrain,YTrain,layers,options);

%% Calculate Prediction 
% use command "predict"
Ypred = predict(trainedNet,XTest);

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR


Pred_rmse = rmse(Ypred(),single(YTest()));

% Anzeigen einiger Ergebnisse für visuelle Kontrolle 
k=0;
for i=1:10
    k = k+1;
    subplot(10,3,k), imshow(XTest(:,:,:,i)),title('Input')
    k = k+1;
    subplot(10,3,k), imshow(YTest(:,:,:,i)),title('Output')
    k = k+1;
    subplot(10,3,k), imshow(Ypred(:,:,:,i)),title('Output Prediction')
end


ypredDim = size(Ypred);


for i = 1 : ypredDim(4)
    
    Pred_ssim(i) = ssim(Ypred(:,:,:, i),single(YTest(:,:,:,i)));
    Pred_psnr(i) = psnr(Ypred(:,:,:, i),single(YTest(:,:,:,i)));    
    Pred_corr(:,:,1,i) = xcorr2(Ypred(:,:,:, 1),single(YTest(:,:,:,1)));

end



%% Boxplots for step 6 of instructions

%% Step 7: create Neural Network Layergraph U-Net

layers = unetLayers([I_px I_px 1],2,'encoderDepth',3);
finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name','Final-ConvolutionLayer');
layers = replaceLayer(layers,'Final-ConvolutionalLayer',finalConvLayer);
layers = removeLayers(layers,'Softmax-Layer');
regLayer = regressionLayer('Name','Reg-Layer');
layers = replaceLayer(layers,'Segmentation-Layer',regLayer);
layers = connectLayers(layers,'Final-ConvolutionLayer','Reg-Layer');

analyzeNetwork(layers)


%% Boxplots for step 8 of instructions
