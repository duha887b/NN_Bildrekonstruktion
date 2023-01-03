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

options.MiniBatchSize = 64;

options.MaxEpochs = 200;

options.ValidationFrequency = 150;

options.InitialLearnRate = 0.001;

options.ExecutionEnvironment = 'auto';

options.ValidationData = {XValid, YValid};

options.Plots = 'training-progress';

options.OutputNetwork = 'best-validation-loss';

options.ValidationPatience = 20;

%% training using "trainNetwork"

mlp_aug = trainNetwork(XTrain,YTrain,layers,options);

load("DATA_MMF_16.mat")
mlp = trainNetwork(XTrain,YTrain,layers,options);


%% Calculate Prediction 
% use command "predict"
Ypred = predict(mlp,XTest);
Ypred_aug = predict(mlp_aug,XTest);

%% Aufgabe 3 Protokoll

% RSME pro Bild und Durchschnitt
Pred_rmse_tmp = rmse(Ypred(),single(YTest()),[1 2]);

for i = 1 : size(Ypred,4)
     
    Pred_ssim(i) = ssim(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_psnr(i) = psnr(Ypred(:,:,1, i),single(YTest(:,:,1,i)));    
    Pred_corr(i) = corr2(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_rmse(i) = Pred_rmse_tmp(:,:,1,i);
end

Pred_rmse_tmp = rmse(Ypred_aug(),single(YTest()),[1 2]);

for i = 1 : size(Ypred_aug,4)
     
    Pred_ssim_aug(i) = ssim(Ypred_aug(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_psnr_aug(i) = psnr(Ypred_aug(:,:,1, i),single(YTest(:,:,1,i)));    
    Pred_corr_aug(i) = corr2(Ypred_aug(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_rmse_aug(i) = Pred_rmse_tmp(:,:,1,i);
end

figure
subplot(2,2,1), boxchart([Pred_rmse; Pred_rmse_aug]'),title('RMSE'), ylabel('RMSE') ,legend(["1:Original 2:Data Augmentation"]);
subplot(2,2,2), boxchart([Pred_corr; Pred_corr_aug]'),title('Correlation'), ylabel('Correlation') ,legend(["1:Original 2:Data Augmentation"]) ;
subplot(2,2,3), boxchart([Pred_psnr; Pred_psnr_aug]'),title('PSNR'), ylabel('PSNR') ,legend(["1:Original 2:Data Augmentation"]) ;
subplot(2,2,4), boxchart([Pred_ssim; Pred_ssim_aug]'),title('SSIM'), ylabel('SSIM') ,legend(["1:Original 2:Data Augmentation"]) ;


%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%Anzeigen einiger Ergebnisse für visuelle Kontrolle 
figure
k=0;
for i=1:10
    k = k+1;
    subplot(10,3,k), imshow(XTest(:,:,:,i),[0 255]),title('Input')
    k = k+1;
    subplot(10,3,k), imshow(YTest(:,:,:,i),[0 255]),title('Output')
    k = k+1;
    subplot(10,3,k), imshow(Ypred(:,:,:,i),[0 255]),title('Output Prediction')
end


ypredDim = size(Ypred);

% RSME pro Bild und Durchschnitt
Pred_rmse_tmp = rmse(Ypred(),single(YTest()),[1 2]);
tmp = 0;
for i=1:size(Pred_rmse_tmp,4)
    tmp = tmp + Pred_rmse_tmp(:,:,1,i);
end
Pred_rmse_d = tmp/size(Pred_rmse_tmp,4);

for i = 1 : ypredDim(4)
     
    Pred_ssim(i) = ssim(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_psnr(i) = psnr(Ypred(:,:,1, i),single(YTest(:,:,1,i)));    
    Pred_corr(i) = corr2(Ypred(:,:,1, i),single(YTest(:,:,1,i)));
    Pred_rmse(i) = Pred_rmse_tmp(:,:,1,i);
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

figure
subplot(2,2,1), boxchart(Pred_rmse),title('RMSE') ;
subplot(2,2,2), boxchart(Pred_corr),title('Correlation');
subplot(2,2,3), boxchart(Pred_ssim),title('SSIM');
subplot(2,2,4), boxchart(Pred_psnr),title('PSNR');
        

%% Step 7: create Neural Network Layergraph U-Net

load("DATA_MMF_16_aug.mat")

Ulayers = unetLayers([I_px I_px 1],2,'encoderDepth',3);
 
finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name','Final-ConvolutionLayer');
Ulayers = replaceLayer(Ulayers,'Final-ConvolutionLayer',finalConvLayer);

Ulayers = removeLayers(Ulayers,'Softmax-Layer');

regLayer = regressionLayer('Name','Reg-Layer');
Ulayers = replaceLayer(Ulayers,'Segmentation-Layer',regLayer);

Ulayers = connectLayers(Ulayers,'Final-ConvolutionLayer','Reg-Layer');

%analyzeNetwork(layers)

unet = trainNetwork(XTrain,YTrain,Ulayers,options);

%% Evaluate Network
UYpred = predict(unet,XTest);

figure
k=0;
for i=1:10
    k = k+1;
    subplot(10,3,k), imshow(XTest(:,:,:,i),[0 255]),title('Input')
    k = k+1;
    subplot(10,3,k), imshow(YTest(:,:,:,i),[0 255]),title('Output')
    k = k+1;
    subplot(10,3,k), imshow(UYpred(:,:,:,i),[0 255]),title('Output Prediction');

end

ypredDim = size(UYpred);

% RSME pro Bild und Durchschnitt
Pred_rmse_tmp = rmse(UYpred(),single(YTest()),[1 2]);
tmp = 0;
for i=1:size(Pred_rmse_tmp,4)
    tmp = tmp + Pred_rmse_tmp(:,:,1,i);
end
UPred_rmse_d = tmp/size(Pred_rmse_tmp,4);

for i = 1 : ypredDim(4)
     
    UPred_ssim(i) = ssim(UYpred(:,:,1, i),single(YTest(:,:,1,i)));
    UPred_psnr(i) = psnr(UYpred(:,:,1, i),single(YTest(:,:,1,i)));    
    UPred_corr(i) = corr2(UYpred(:,:,1, i),single(YTest(:,:,1,i)));
    UPred_rmse(i) = Pred_rmse_tmp(:,:,1,i);
end

%Durchsnitt SSIM, PSNR, CORR
tmp=0;
for i=1:size(UPred_ssim,2)
    tmp = tmp + UPred_ssim(1,i);
end
UPred_ssim_d = tmp/size(UPred_ssim,2);

tmp=0;
for i=1:size(UPred_psnr,2)
    tmp = tmp + UPred_psnr(1,i);
end
UPred_psnr_d = tmp/size(UPred_psnr,2);

tmp=0;
for i=1:size(UPred_corr,2)
    tmp = tmp + UPred_corr(1,i);
end
UPred_corr_d = tmp/size(UPred_corr,2);

%% Boxplots for step 8 of instructions
figure
subplot(2,2,1), boxchart([UPred_rmse; Pred_rmse_aug]'),title('RMSE'), ylabel('RMSE') ,legend(["1:Unet 2:MLP"]);
subplot(2,2,2), boxchart([UPred_corr; Pred_corr_aug]'),title('Correlation'), ylabel('Correlation') ,legend(["1:Unet 2:MLP"]) ;
subplot(2,2,3), boxchart([UPred_psnr; Pred_psnr_aug]'),title('PSNR'), ylabel('PSNR') ,legend(["1:Unet 2:MLP"]) ;
subplot(2,2,4), boxchart([UPred_ssim; Pred_ssim_aug]'),title('SSIM'), ylabel('SSIM') ,legend(["1:Unet 2:MLP"]) ;

        