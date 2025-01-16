
% signal = load('Dog_Sig.mat');
% signal = signal.SegFreq;
% compareAudioSegments(signal)
% 
function [word, corrCoeff] = compareAudioSegmentsCross(SegFreq, noise)
    fs = 16e3;
    fplot = -4.8e4:10:4.8e4;
    sea = load('Sea_1_1620_Lab.mat');
    jaguar = load('Jaguar_1_1620_Lab.mat');
    % cat = load('Cat_1_1620_Home.mat');
    elephant = load('Elephant_1_1620_Lab.mat');
    fish = load('Fish_1_1620_Lab.mat');
    mouse = load('Mouse_1_1620_Lab.mat');
    downsampleFactor = 2;

    % avgNoise = mean(noise(:));
    % SegFreq = SegFreq./(SegFreq+avgNoise); 
    sea = sea.sSegFreq;
    % dog = dog.sSegFreq; %SpeechSegsF; 
    jaguar = jaguar.sSegFreq;
    % cat = cat.sSegFreq; %SpeechSegsF; 
    elephant = elephant.sSegFreq; %SpeechSegsF; 
    fish = fish.sSegFreq; %SpeechSegsF; 
    mouse = mouse.sSegFreq; %SpeechSegsF;
    dict = {sea, jaguar, elephant, fish, mouse};
    crossCorrs = cell(size(dict));
    
    corrCoeff = zeros([5, 5]);
    %SegFreq = (SegFreq-mean(SegFreq(:)))/std(SegFreq(:));
    % SegFreqDS = downsample(SegFreq, downsampleFactor);
    for i = 1:5
        cur_word = dict{i};
        for j = 1
            dictSignal = cur_word{j};
            % dictSignal = reshape(dictSignal, 9600, []);
            %dictSignal = (dictSignal-mean(dictSignal(:)))/std(dictSignal(:));
            % dictSignalDS = downsample(dictSignal, downsampleFactor, 'absolute');
             % [dist, path] = dynamicTimeWarp(SegFreq, dictSignal);


             % corrCoeff(i,j) = 1 / (1 + dist);
            % fprintf('Correlation between segment signal and segment %d (Frequency Domain): %.4f\n', i+j,  corrCoeff(i,j));
            % maxLen = max(size(SegFreq, 2), size(dictSignal, 2));
            % aligned_SegFreq = zeros(size(SegFreq,1), maxLen);  
            % aligned_dictSig = zeros(size(SegFreq,1), maxLen);
            % 
            % for i = 1:size(SegFreq,1)
            %     [dist, x, y] = dtw(SegFreq(i,:), dictSignal(i,:), 'absolute');
            % 
            %     aligned_SegFreq(i, :) = SegFreq(i, x);  
            %     aligned_dictSig(i, :) = dictSignal(i, y);
            % end
            % 
            % corrMatrix= corrcoef([aligned_SegFreq, aligned_dictSig]);
            % corrCoeff(i,1) = corrMatrix(1, 2);
            % if size(SegFreqDS, 2) < maxLen
            %     X1 = padarray(SegFreqDS, [0, maxLen - size(SegFreq, 2)], 0, 'post');
            % else
            %     X1 = SegFreqDS;
            % end
            % 
            % if size(dictSignal, 2) < maxLen
            %     X2 = padarray(dictSignalDS, [0, maxLen - size(dictSignal, 2)], 0, 'post');
            % else
            %     X2 = dictSignalDS;
            % end
            % 
            % corrMatrix = xcorr2(X1, X2);
            % % Find the maximum correlation coefficient
            % currentMaxCorr = max(corrMatrix(:));
            % 
            % if currentMaxCorr > maxCorr
            %     maxCorr = currentMaxCorr;
            %     word = words{i};
            % end
            %[dist, ix, iy] = dtw(X1, X2, 'Absolute', 'Radius', 10);
            % corrCoeff(i) = 1 / (1 + dist);
            % figure;
            % subplot(2, 1, 1);
            % imagesc(maxLen,fplot,X1);
            % title('X1 - Padded SegFreq');
            % xlabel('Time');
            % ylabel('Frequency');
            % colorbar;
            % clim([0 1])
            % ylim([-5000 5000])
            % 
            % subplot(2, 1, 2);
            % imagesc(maxLen,fplot,X2);
            % title('X2 - Padded dictSignal');
            % xlabel('Time');
            % ylabel('Frequency');
            % colorbar;
            % clim([0 1])
            % ylim([-5000 5000])
            % 
            % X1 = X1(:);
            % X2 = X2(:);

            [crossCorr, ~] = xcorr(SegFreq(:), dictSignal(:));
            normDooDad = norm(SegFreq)*norm(dictSignal);
            crossCorr = crossCorr/normDooDad;
            corrCoeff(i) = max(crossCorr);

      
            % bestLag = lags(maxIdx);
            % 
            % if bestLag > 0
            %     X1 = SegFreq(bestLag+1:end);
            %     X2 = dictSignal(1:min(length(dictSignal), length(X1)));
            %     X1 = X1(1:length(X2));
            % else
            %     X2 = SegFreq(-bestLag+1:end);
            %     X1 = SegFreq(1:min(length(SegFreq), length(X2)));
            %     X2 = X2(1:length(X1));
            % end
            % 
            % 
            % corrMatrix= corrcoef(X1', X2');
            % corrCoeff(i,1) = corrMatrix(1, 2);
            % fprintf('Correlation between segment signal and segment %d (Frequency Domain): %.4f\n', i, corrCoeff(i,1));
            % if corrMatrix(1,2) >.85
            %     break;
            % end
    % figure;
    % subplot(2,1,1);
    % plot(SegTime);  
    % title('SegFreq - Time Domain');
    % 
    % subplot(2,1,2);
    % plot(real(ifft(dictSignal))); 
    % title('DictSignal - Time Domain');
    % 
    % figure;
    % subplot(2,1,1);
    % plot(abs(SegFreq)); 
    % title('SegFreq - Frequency Domain');
    % 
    % subplot(2,1,2);
    % plot(abs(dictSignal));  
    % title('DictSignal - Frequency Domain');

        end
    end

    [maxCorr, linearIdx] = max(corrCoeff(:));
    [maxRow, ~] = ind2sub(size(corrCoeff), linearIdx);
    if maxCorr > .01
        switch maxRow
            case 1
                word = 'sea';
            case 2
                word = 'jaguar';
            case 3
                word = 'elephant';
            case 4
                word = 'fish';
            case 5
                word = 'mouse';
        end
     else
         word = 'Not a recognized word';
     end

    fprintf('The word being said is: %s\n', word);
end