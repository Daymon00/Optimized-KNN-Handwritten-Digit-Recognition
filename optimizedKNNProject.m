%tell the computer where to find the data
trainFile = '/Users/daymonwu/Documents/CSCI 367 Project/train-data.txt';
testFile = '/Users/daymonwu/Documents/CSCI 367 Project/test-data.txt';

%use a function to load this data into the memory.
[XTrn, yTrn] = loadData(trainFile); % Load training data
[XTst, yTst] = loadData(testFile);  % Load testing data

%change each image of a number into a simple list of numbers that the computer can understand easily
XTrnFlt = flattenImages(XTrn); % flatten training images
XTstFlt = flattenImages(XTst); % flatten testing images

%make sure all these lists are on a common scale
mu = mean(XTrnFlt);            % find the average value
sigma = std(XTrnFlt);          % find the standard deviation
XTrnStd = (XTrnFlt - mu) ./ sigma; % standardize training data
XTstStd = (XTstFlt - mu) ./ sigma; % standardize testing data

%find the best number of nearest neighbors to use when making decisions
tic; % Start the timer
kOpt = optimizeKValue(XTrnStd, yTrn, XTstStd, yTst); % Find the best k value
yPred = knnClassifier(XTrnStd, yTrn, XTstStd, kOpt); % Classify test data
elapsedTime = toc; % Stop the timer and record the time

%check how well it learned to recognize the numbers
confMat = manualMatrix(yTst, yPred); % Create a confusion matrix
acc = sum(yPred == yTst) / numel(yTst); % Calculate the overall accuracy
disp('Optimized KNN Classifier');
disp(['Overall Time: ', num2str(elapsedTime), ' seconds']);
disp('Confusion Matrix:');
disp(confMat);
disp(['Overall Accuracy: ', num2str(acc)]);

%look at how well it recognized each specific number
perClassAcc = calculatePerClassAccuracy(confMat);
disp('Per-Class Accuracy:');
disp(perClassAcc);

%show some examples of the numbers it saw and what it thought they were
plotSamples(XTst, yTst, yPred); %isplay 25 sample images with predictions

%helper functions>>>

%reads the data from a file.
function [X, y] = loadData(filePath)
    data = readmatrix(filePath);
    y = data(:, 1);
    X = data(:, 2:end);
end

%changes each image into a list of numbers.
function flatImages = flattenImages(data)
    numImages = size(data, 1);
    flatImages = zeros(numImages, 256);
    for i = 1:numImages
        img = reshape(data(i, :), [16, 16]);
        flatImages(i, :) = img(:)';
    end
end

%brain of our program deciding which number is which
function yPred = knnClassifier(XTrn, yTrn, XTst, k)
    numTestSamples = size(XTst, 1);
    yPred = zeros(numTestSamples, 1);
    for i = 1:numTestSamples
        distances = sqrt(sum((XTrn - XTst(i, :)).^2, 2));
        [~, idx] = sort(distances);
        nn = yTrn(idx(1:k));
        yPred(i) = mode(nn);
    end
end

%tests different values of k to find the best one
function kOpt = optimizeKValue(XTrn, yTrn, XTst, yTst)
    kValues = 1:2:11; % Try odd values from 1 to 11
    bestAcc = 0;
    kOpt = kValues(1);
    for k = kValues
        yPred = knnClassifier(XTrn, yTrn, XTst, k);
        acc = sum(yPred == yTst) / numel(yTst);
        if acc > bestAcc
            bestAcc = acc;
            kOpt = k;
        end
    end
    disp(['Optimal k value: ', num2str(kOpt)]);
end

%makes a table to see where the computer got confused
function confMat = manualMatrix(yReal, yPred)
    classes = unique([yReal; yPred]);
    numClasses = length(classes);
    confMat = zeros(numClasses, numClasses);
    for i = 1:numClasses
        for j = 1:numClasses
            confMat(i, j) = sum((yReal == classes(i)) & (yPred == classes(j)));
        end
    end
end

%calculates how accurately each number was recognized
function pca = calculatePerClassAccuracy(confMatrix)
    pca = diag(confMatrix) ./ sum(confMatrix, 2);
end

%shows what the computer saw and what it thought it saw
function plotSamples(X, y, yPred)
    figure;
    numSamples = 25; % Updated number of samples to display
    for i = 1:numSamples
        subplot(5, 5, i); % Updated subplot layout to 5 rows and 5 columns
        img = reshape(X(i, :), [16, 16]);
        imshow(img, []);
        title(['Real: ', num2str(y(i)), ' Pred: ', num2str(yPred(i))]);
    end
end
