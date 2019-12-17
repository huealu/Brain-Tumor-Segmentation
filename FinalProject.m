%% ----------------------------------------------------------------------%%
%                       Image Processing - COSC 6324                      %
%                               Fall 2019                                 %
%                             Final Project                               %
%  Brain Tumor Segmentation using Mophological Reconstruction and K-Means %
%                       Clustering on MRI image                           %
%                   By Thi Dinh and Collin Breedlove                      %
%%-----------------------------------------------------------------------%%
close all;
clear all;
clc;
%% Read image into program
inputImg = imread('Data\Y1.jpg');
%% -------------------------Pre-processing stage-------------------------%%
% Pre-processing process includes:
%       1. Scale and converted the input image to grayscale for further 
%       processing
%       2. Removed noise by applying median filter
%       3. Modified contrast of the image using histogram equalization
%       4. Sharped the image using Gaussian high pass filter on frequency
%       domain
%       5. Performed Skull Stripping

%%                      1. Convert rgb image to grayscale
brain = imresize(inputImg, [256 256]);
[rows columns colorChanel] = size(brain);
if colorChanel>1
    gray = rgb2gray(brain);
else
    gray = brain;
end;

figure('Name','Preprocessing Image') 
subplot(2,2,1)
imshow(gray); title('Original Image'); axis on; 

% Initialize a counting time 
time = zeros(1,15);
%%                  2. Apply median filter to smooth the image
tic;

filtering = medfilt2(gray);
subplot(2,2,2)
imshow(filtering); title('After median filtering');axis on;

%                   3. Apply histogram equalization on image
% Draw out the histogram of the image
[pixelCount, grayLevel] = imhist(filtering);
subplot(2,2,3); 
bar(grayLevel, pixelCount);
grid on; axis on;
title('Histogram of image'); xlim([0 grayLevel(end)]);

% Perform adapt histogram equalization
histEqual = adapthisteq(filtering,'NumTiles',[8 8],'ClipLimit',0.0005);
subplot(2,2,4);
imshow(histEqual); title('After histogram equalization');
axis on;

time(1) = toc;

%%              4. Apply Gaussian High Pass Filter on Frequency Domain
tic;

dou_image = double(histEqual);
[row,column] = size(histEqual); % Obtain the size of the image
new_row = 2 * row;
new_column = 2 * column;

% Double size of the image
pad_image=zeros((new_row),(new_column));
cen_image=zeros((new_row),(new_column));

% Create padding of image
for i = 1 : row
    for j = 1 : column
        pad_image(i,j) = dou_image(i,j);
    end
end

% Translate to center of frequency rectangle
for i = 1 : row
    for j = 1 : column
        cen_image(i,j) = pad_image(i,j)*((-1)^(i+j));
    end
end

% Perform 2D Fast Fourier Transformation to transform to frequency domain
fou_image = fft2(cen_image);

% Performing Gaussian High Pass Filter
n = 1;          % Order for butterworth filter
thresh = 10;    % Cutoff radius in frequency domain for filters
gau_image = gau_high_fil(fou_image, thresh);  % Apply Gaussian high pass filter

%Inverse 2D Fast Fourier Transformation
inv_fou_image = ifft2(gau_image);

% Translate the image back to the corner
for i = 1:new_row
    for j = 1:new_column
        inv_fou_image(i,j) = inv_fou_image(i,j)*((-1)^(i+j));
    end
end
% Remove the padding to get sharp image
for i = 1:row
    for j = 1:column
        sha_image(i,j) = inv_fou_image(i,j);
    end
end
% Remove/convert value of the pixel into real number
sha_image = real(sha_image);
sha_image = uint8(sha_image);

figure('Name','Fourier Transform'), 
imshow(sha_image);title('Image Sharpening'); axis on;

time(2) = toc;

%%                         5. Perform Skull Stripping
%           a. Apply Otsu binarization - global threshold the image
% Find the approximate threshhold then binarize the image

tic;

level = graythresh(sha_image);
binaryImage = imbinarize(sha_image, 'global');
figure('Name','Otsu Binarization')

% Remove small objects which is smaller than 0.5% of the image
binaryImage = bwareaopen(binaryImage,round(256*256/100*.5));
imshow(binaryImage);title ('After Otsu binarization');  axis on;

%       b. Dilate the image then fill in the hole inside the object
binaryImage = bwareaopen(binaryImage,10);
di1 = imdilate(binaryImage,true(5));
labeledImage = bwlabel(di1);		% Assign label ID numbers to all blobs.
ab = ismember(labeledImage, 1);     % Use to extract blob 1.
dilutionImage = imdilate(ab,true(5));

figure('Name','Skull Stripping')
subplot(2,2,1);imshow(dilutionImage); 
title('Dilution Gradient Mask'); axis on;

% Fill the holes in image
fillImage = imfill(dilutionImage, 'holes');
subplot(2,2,2);
imshow(fillImage); title('Filling the holes of binary image'); axis on;

%                       c. Erode the image using disk
se1 = strel('disk', 15);
erodeImage = imerode(fillImage,se1);
subplot(2,2,3);
imshow(erodeImage); title('Eroded Binary image'); axis on;

%               d. Skull stripping extraction on the original image
stripImage = immultiply(gray, erodeImage);
subplot(2,2,4)
imshow(stripImage); title('Skull Stripping image');axis on;

time(3) = toc;
%%------------------------ End Preprocessing Image ----------------------%%


%% -------------------------Segmentation stage---------------------------%%
% - Assume that the area with intensity is greaeter than 205 in grayscale
% image is the tumor 
% - Since the image without skull is ready, the next stage is segmentation
% to find the objects on the preprocessed image
% - The process includes:
%   1. Find tumor using thresholding technique
%   2. Extract the tumor areas by applying:
%       + Dilation
%       + Fill in the hole
%       + Erode
%       + Removing small object (<0.5% of the image)

%%                   1. Apply thresholding to find tumor
% Apply the histogram equalization on the image without skull
tic;

stripImage(stripImage<50) = 0;
stripImage = adapthisteq(stripImage,'NumTiles',[8 8],'ClipLimit',0.005);

% Binarized the image with threshold of 205
binarySkull = stripImage > 205;
figure('Name','Segmentation')
subplot(2,2,1);
imshow(binarySkull); title('Initial binary image'); axis on;

%%                       2. Extract the tumor areas 
% Dilate the histogram equalized image (without skull)
di1 = imdilate(binarySkull,true(10));

% Fill the holes inside the objects
binaryTumor_dil = imfill(di1, 'holes');

% Erode the image using disk
se11 = strel('disk', 5);
binaryTumor_erode = imerode(binaryTumor_dil,se11);

% Remove small objects to extract the largest blob assumed as a tumor
binaryTumor = bwareaopen(binaryTumor_erode,round(256*256/100*.5));
subplot(2,2,2);
imshow(binaryTumor); title('Binary Tumor'); axis on;

% Extract boundaries of tumor on the original image
subplot(2,2,3); imshow(gray); axis on;
title('Tumor in brain');
hold on;
boundaryTumor = bwboundaries(binaryTumor);
boundNum = size(boundaryTumor,1);
for i = 1:boundNum
    bound = boundaryTumor{i};
    plot(bound(:,2), bound(:,1), 'r', 'LineWidth', 2)
end;
hold off;

% Overlay the tumor in brain
subplot(2,2,4);
imshow(stripImage); axis on; title('Tumor region in brain');
hold on;
sizeTumor = size(binaryTumor);
overlayImg = cat(3, ones(sizeTumor), zeros(sizeTumor), zeros(sizeTumor));
redImg = imshow(overlayImg);
hold off;
alpha = 0.3*double(binaryTumor);
set(redImg, 'AlphaData', alpha); axis on;

time(4) = toc;
%% -------------------------- End Segmentation Stage --------------------%%


%% ------------------------ Morphological Stage ------------------------%%
% The process of Morphological includes:
%   1. Applying Opening and Closing for image
%   2. Create mask
%   3. Extract boundaries of tumor
%   4. Overlay the tumor in brain

tic; 

binarySkull = double(binaryTumor(:,:));
stripBrain1 = imbinarize(binarySkull, 'global');
stripBrain = immultiply(stripBrain1, erodeImage);
figure('Name','Morphological')
subplot(2,3,1);
imshow(stripBrain1); title('Brain Stripping image'); axis on;

%%               1. Morphological opening and closing
se2 = strel('disk', 3, 0);
se3 = strel('square', 3);
openedImg = imopen(stripBrain, se2);
subplot(2,3,2); imshow(openedImg); title('Applying Morphological Opening');
axis on; hold on;
closedImg = imclose(openedImg, se3);
subplot(2,3,3); imshow(closedImg); title('Applying Morphological Closing');
axis on;

%%                           2. Create mask 
binTumor = bwareafilt(closedImg,8);
subplot(2,3,4);
imshow(binTumor); title('Binary Tumor'); axis on;

%%                    3. Extract boundaries of tumor
subplot(2,3,5); imshow(stripImage); axis on;
title('Tumor in brain');
hold on;
boundaryTumor = bwboundaries(binTumor);
boundNum = size(boundaryTumor,1);
for i = 1:boundNum
    bound = boundaryTumor{i};
    plot(bound(:,2), bound(:,1), 'r', 'LineWidth', 2)
end;
hold off;

%%                       4. Overlay the tumor in brain
subplot(2,3,6);
imshow(brain); title('Tumor region in brain');
hold on; 
sizeTumor = size(binTumor);
overlayImg = cat(3, ones(sizeTumor), zeros(sizeTumor), zeros(sizeTumor));
redImg = imshow(overlayImg);
hold off;
alpha = 0.3*double(binTumor);
set(redImg, 'AlphaData', alpha); axis on;

time(5) = toc;
%% -------------------- End Morphological Stage ------------------------%%


%% --------------Apply KMeans for extracting segment--------------------%%
% Apply K-Means clustering

numberCluster = 3;      % Initialize number of clusters

stripImage1 = immultiply(histEqual, erodeImage);

for i = 3:numberCluster

    tic;

    grayScale = double(stripImage1(:));

    [clusterInx, clusterCent] = kmeans(grayScale, i, 'distance',...
                                    'sqEuclidean', 'Replicates', 2);
    % Reshape the clusterInx to matrix
    labeledImg = reshape(clusterInx, rows, columns);

    
    cap1 = sprintf('K-Means with %d clusters',i); 
    figure('Name',cap1);
    subplot(1,2,1); imshow(labeledImg,[]);
    title(cap1); axis on;

    % Label each cluster by different color
    labelColor = label2rgb(labeledImg, 'hsv', 'k', 'shuffle');
    subplot(1,2,2); imshow(labelColor);  
    title('Colored clustering'); axis on;

    segmented_image = cell(1, i);
    
    % Show the segmented images of each cluster
    figure('Name','Segmented Images')
    for k = 1:i
        grayImg = stripImage;
        grayImg(labeledImg ~= k) = 0;
        % Global threshold the segmented images
        segmented_image{k} = imbinarize(grayImg, 'global');
        subplot(2,3,k);
        caption = sprintf('Object in Cluster %d', k);
        imshow(segmented_image{k});title(caption);
        axis on;
    end

    time(6) = toc;

    tic;
    
    % Apply Discrete Wavelet Transform
    inDwt = double(labeledImg(:,:));
    segImage = imbinarize(inDwt, 'global');
    
    % 1st-level decomposition DWT
    [cA1, cH1, cV1, cD1] = dwt2(inDwt, 'db3');
    
    % 2nd-level decomposition DWT
    [cA2, cH2, cV2, cD2] = dwt2(cA1, 'db3');

    % 3rd-level decomposition DWT
    [cA3, cH3, cV3, cD3] = dwt2(cA2, 'db3');
    
    figure('Name','Result after applying Discrete Wavelet Transformation')
    subplot(2,2,1); imagesc(cA3);
    colormap gray; title('Approximation');axis on;
    subplot(2,2,2); imagesc(cH3);
    colormap gray; title('Horizontal');axis on;
    subplot(2,2,3); imagesc(cV3);
    colormap gray; title('Vertical');axis on;
    subplot(2,2,4); imagesc(cD3);
    colormap gray; title('Diagonal');axis on;

    time(7) = toc;

    tic;
    
    % Apply PCA 
    outputDwt = [cA3 cH3 cV3 cD3];

    [coeff,score,latent] = pca(outputDwt);

    % Gray-level co-occurence matrix
    co_matrix = graycomatrix(coeff);

    % Inverse Hotelling Tranform 
    numberFeature = 3;

    [m n] = size(coeff);
    co_feature = coeff(:,1:numberFeature);
    temp = zeros(m,n-numberFeature);
    feature_matrix = [co_feature temp];

    inverse_pca = score*feature_matrix';

    [a1 b1] = size(cA3);
    [a2 b2] = size(cH3);
    [a3 b3] = size(cV3);
    [a4 b4] = size(cD3);

    figure('Name', 'Result after using PCA')
    Inverse_cA3 = inverse_pca(:, 1:b1);
    subplot(2,2,1); imshow(Inverse_cA3);axis on;

    sum_b1b2 = b1+b2;
    Inverse_cH3 = inverse_pca(:, b1+1:sum_b1b2);
    subplot(2,2,2); imshow(Inverse_cH3);axis on;

    sum_b1b2b3 = b1+b2+b3;
    Inverse_cV3 = inverse_pca(:, (sum_b1b2+1):sum_b1b2b3);
    subplot(2,2,3); imshow(Inverse_cV3);axis on;

    Inverse_cD3 = inverse_pca(:, (sum_b1b2b3):(sum_b1b2b3+b4));
    subplot(2,2,4); imshow(Inverse_cD3);axis on;

    time(8) = toc;

    tic;
    % Apply SVM - Source from Manu
    % <https://www.mathworks.com/matlabcentral/fileexchange/55107-brain-mri-tumor-detection-and-classification>
    stats = graycoprops(co_matrix);
    Contrast = stats.Contrast;
    Correlation = stats.Correlation;
    Energy = stats.Energy;
    Homogeneity = stats.Homogeneity;
    Mean = mean2(coeff);
    Standard_Deviation = std2(coeff);
    Entropy = entropy(coeff);
    RMS = mean2(rms(coeff));
    Variance = mean2(var(double(coeff)));
    a = sum(double(coeff(:)));
    Smoothness = 1-(1/(1+a));
    Kurtosis = kurtosis(double(coeff(:)));
    Skewness = skewness(double(coeff(:)));
    
    % Inverse Difference Movement
    m = size(coeff,1);
    n = size(coeff,2);
    in_diff = 0;
    for i = 1:m
        for j = 1:n
            temp = coeff(i,j)./(1+(i-j).^2);
            in_diff = in_diff+temp;
        end
    end
    IDM = double(in_diff);


    feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation,...
        Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

    load Trainset.mat
    xdata = meas;
    group = label;
    svmStruct1 = fitcsvm(xdata,group,'KernelFunction', 'rbf');
    [imgLabel, score_svm] = predict(svmStruct1,feat);

    data1   = [meas(:,1), meas(:,2)];
    newfeat = [feat(:,1),feat(:,2)];

    Answer = ['The image shows: ', imgLabel];
    time(9) = toc;
end

disp('The processing time of each stage is:');
disp('Pre-processing time: ');

for j=1:4
    disp(time(j))
end;
disp('Processing time of segmetation by morphological reconstruction: ');
disp(time(5));
disp('Processing time of segmetation by K-Means:');
disp(time(6));
disp('Processing time of DWT: ');
disp(time(7));
disp('Processing time of PCA: ');
disp(time(8));
disp('Classification procssing time: ');
disp(time(9));
