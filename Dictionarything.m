clear all
close all

device = 'Sound Blaster G3';
fs = 16e3;
frameRate = 20; %5 plot 25 no plot
timeLim1 = 3;
timeLim2 = 60;
frameSize = floor(fs/frameRate);
pool = gcp;

userInputs = {};
numInputs = input('Enter the number of inputs: ');
for i = 1:numInputs
        userInput = input('Enter a string: ', 's');
        userInputs{i} = userInput; 
end


[noiseRaw, noiseSTFT, noisePow, noiseTime, noiseFreq] = audio_analysis_function(fs, frameRate, device, timeLim1);
[sigRaw, sigSTFT, sigPow, sigTime, sigFreq, fftData, sSegTime, sSegFreq, coeffs, guesses, trueNegatives] = analyzeAudioFunction(fs, frameRate, device, timeLim2, noisePow, noiseSTFT);

%% falsdhkf
guesses = guesses(1:20);
% userInputs = userInputs{1:19};
if exist('userInputs', 'var') && length(userInputs) == length(guesses)
        [confMat, order] = confusionmat(userInputs, guesses);
        disp('Confusion Matrix:');
        disp(confMat);
        disp('Order of words:');
        disp(order);
        confusionChart = confusionchart(userInputs, guesses);
        title('Confusion Matrix');
else
        disp('Expected words and spoken word results do not match in length.');
end


if length(guesses) > length(userInputs)
    falsePositives = length(guesses)-length(userInputs);
    truePositives = length(guesses)-falsePositives;
    falseNegatives = 0;
else 
    falseNegatives = length(userInputs)-length(guesses); 
    truePositives = length(userInputs)-trueNegatives;
    falsePositives = 0;
end

TPR = truePositives./(truePositives+falseNegatives);
FPR = falsePositives./(falsePositives+trueNegatives);

FPR = [.5714 .2 0 0 0 0 0];
TPR = [.2 .8 .8 .6 .6 .6 .4];
x = 0:1;
y = x;
figure;
hold on
plot(FPR, TPR, '-o');
plot(x, y)
hold off
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve');
legend('ROC Curve', 'Random Guess Line')
grid on;
xlim([0 1])
ylim([0 1])

% save('GroundTruthDTW2.m', userInputs)
% save('GuessesDTW2.m', guesses)

%% Identify Speech Segments
% overlap = 1/2;
% hopSize = frameRate * (1 - overlap);
% frameTime = (frameSize+hopSize) / fs;
% colSig = size(sigPow, 2);
% colNoise = size(noisePow, 2);
% repeatFactor = ceil(colSig / colNoise); 
% noisePowRepeated = repmat(noisePow, 2, repeatFactor);
% noisePow = noisePowRepeated(:, 1:colSig);
% t = (0:colSig-1) * frameTime;
% 
% avgPow_S = zeros(1,colSig);
% avgPow_N = zeros(1,colSig);
% 
% for i = 1:colSig
%     avgPow_S(i) = sum(sigPow(:,i))/size(sigPow,1);
%     avgPow_N(i) = sum(noisePow(:,i))/size(noisePow,1);
% end
% SNR = 20*log10(avgPow_S./avgPow_N); 
% plotFrames = 1:length(avgPow_S);
% SNR_filt = movmedian(SNR, 3);
% dSNR = diff(SNR);
% dSNR_filt = movmedian(dSNR, 3);
% filt_dSNR = diff(SNR_filt);
% 
% % figure;
% % subplot(2,1,1)
% % plot(dSNR);
% % title('Derivative of SNR');
% % xlabel('Frame Index');
% % ylabel('dSNR/dt');
% % grid on;
% 
% % subplot(2,1,2)
% % plot(dSNR_filt);
% % title('Filtered of derivative of SNR');
% % xlabel('Frame Index');
% % ylabel('dSNR/dt');
% % grid on;
% % 
% % 
% % figure;
% % subplot(2,1,1)
% % plot(SNR_filt);
% % title('Filtered SNR');
% % xlabel('Frame Index');
% % ylabel('dSNR/dt');
% % grid on;
% % 
% % subplot(2,1,2)
% % plot(filt_dSNR);
% % title('Derivative of filtered SNR');
% % xlabel('Frame Index');
% % ylabel('dSNR/dt');
% % grid on;
% % 
% % filt_dSNR_filt = movmedian(filt_dSNR,3);
% % figure
% % plot(filt_dSNR_filt);
% % title('filtered Derivative of filtered SNR');
% % xlabel('Frame Index');
% % ylabel('dSNR/dt');
% % grid on;
% 
% 
% SNR_thresh = 10;
% speechFrames = find(SNR_filt(2:end) > SNR_thresh);
% speechSegs = [];
% 
% % wordStarts = [];
% % wordEnds = [];
% % inWord = false;
% 
% % for i = 3:length(filt_dSNR_filt)
% %     if ~inWord
% % 
% %         if filt_dSNR_filt(i) >= filt_dSNR_filt(i-1) && filt_dSNR_filt(i-1) > filt_dSNR_filt(i-2) && filt_dSNR_filt(i-2) >= 0 && filt_dSNR_filt(i)>=2 && SNR_filt(i) > SNR_thresh
% %             wordStarts = [wordStarts, i-2]; 
% %             inWord = true;
% %             disp(['Word start detected at index: ', num2str(i-2)]);
% %             disp(['dSNR values: ', num2str(filt_dSNR_filt(i-2)), ', ', num2str(filt_dSNR_filt(i-1)), ', ', num2str(filt_dSNR_filt(i))]);
% %         end
% %     else
% %         if filt_dSNR_filt(i) > -.5 && filt_dSNR_filt(i-1) < 0 && filt_dSNR_filt(i-2) < 0
% %             wordEnds = [wordEnds, i];
% %             inWord = false;
% %             disp(['Word end detected at index: ', num2str(i)]);
% %             disp(['dSNR values: ', num2str(filt_dSNR_filt(i-2)), ', ', num2str(filt_dSNR_filt(i-1)), ', ', num2str(filt_dSNR_filt(i))]);
% %         end
% %     end
% % end
% % if ~isempty(wordStarts) && ~isempty(wordEnds)
% %     for j = 1:length(wordStarts)
% %         if j <= length(wordEnds)
% %             speechSegs = [speechSegs; wordStarts(j), wordEnds(j)];
% %         end
% %     end
% % end
% if ~isempty(speechFrames)
%     startIdx = speechFrames(1);
%     for i = 2:length(speechFrames)
%         if speechFrames(i) > speechFrames(i-1) + 5
%             speechSegs = [speechSegs; startIdx, speechFrames(i-1)+3];
%             startIdx = speechFrames(i); 
%         end
%     end
% speechSegs = [speechSegs; startIdx, speechFrames(end)+3];
% end
% 
% speechTimes = zeros(size(speechSegs));
% 
% for i = 1:size(speechSegs, 1)
%     startFrame = speechSegs(i, 1)+1;
%     endFrame = speechSegs(i, 2)+1;
% 
%     startTime = (startFrame - 1) * frameTime;  
%     endTime = (endFrame - 1) * frameTime;  
% 
%     speechTimes(i, :) = [startTime, endTime];
% end
% 
% plotSigRaw = reshape(sigRaw, [], 1);
% t_plot =  (0:length(plotSigRaw)-1) / fs;
% plotSigSTFT = reshape(sigSTFT, [], 1);
% f_plot = (0:length(plotSigSTFT)-1)*fs/length(plotSigSTFT);
% 
% 
% figure;
% plot(t_plot, plotSigRaw); 
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Full Audio Signal vs. Time');
% hold on;
% 
% for i = 1:size(speechTimes, 1)
%     xline(speechTimes(i, 1), '--r', 'Start Speech');
%     xline(speechTimes(i, 2), '--g', 'End Speech');
% end
% 
% hold off;
% 
% figure;
% imagesc(sigTime, sigFreq, 20*log10(abs(sigSTFT).^2));
% axis xy;
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% title('STFT Spectrogram w/ Speech Segments');
% colorbar;
% hold on;
% for i = 1:size(speechTimes, 1)
%     xline(speechTimes(i, 1), '--r', 'Start Speech');
%     xline(speechTimes(i, 2), '--g', 'End Speech');
% end
% hold off;
% 
% figure
% plot(plotFrames, SNR);
% xlabel('Frame Num')
% ylabel('SNR')
% title('SNR vs Frame Number')
% 
% figure
% plot(t, SNR);
% xlabel('Time (s)')
% ylabel('SNR (dB)')
% title('SNR vs Time')
% 
% figure;
% plot(t, SNR_filt);
% hold on;
% for i = 1:size(speechTimes, 1)
%     xline(speechTimes(i, 1), '--r', 'Start Speech');
%     xline(speechTimes(i, 2), '--g', 'End Speech');
% end
% xlabel('Time (s)');
% ylabel('SNR (dB)');
% title('Filtered SNR with Detected Speech Segments');
% hold off;
% 
% figure;
% plot(t, SNR);
% hold on;
% for i = 1:size(speechTimes, 1)
%     xline(speechTimes(i, 1), '--r', 'Start Speech');
%     xline(speechTimes(i, 2), '--g', 'End Speech');
% end
% xlabel('Time (s)');
% ylabel('SNR (dB)');
% title('SNR with Detected Speech Segments');
% hold off;
% 
% % plot(filt_dSNR_filt);
% % hold on;
% % 
% % % Plot vertical lines for wordStarts
% % for k = 1:length(wordStarts)
% %     xline(wordStarts(k), 'g', 'LineWidth', 1.5); 
% % end
% % 
% % % Plot vertical lines for wordEnds
% % for k = 1:length(wordEnds)
% %     xline(wordEnds(k), 'r', 'LineWidth', 1.5); 
% % end
% % 
% % % Add labels and legend
% % xlabel('Frame Index');
% % ylabel('filt_dSNR_filt');
% % title('filt_dSNR_filt with Speech Segment Start and End Times');
% % legend('filt_dSNR_filt', 'Start Times', 'End Times');
% % 
% % hold off;
% %% Correlation Between Segments
% sSegTime = {};
% sSegFreq = {};
% for i = 1:size(speechSegs, 1)
%     startFrame = speechSegs(i, 1);
%     endFrame = speechSegs(i, 2);
% 
%     sigMatrixT = sigRaw(:,startFrame:endFrame);
%     sigVecT = reshape(sigMatrixT,1,[]);
%     sSegTime{i} = sigVecT;
% 
%     sigMatrixF = fftData(:,startFrame:endFrame);
%     sigVecF = reshape(sigMatrixF,1,[]);
%     sSegFreq{i} = sigVecF;
% 
% end
% HPCutoff = 500;
% BPRange = [150 4000];
% [hp_b, hp_a] = butter(4, HPCutoff/(fs/2), 'high');
% [bp_b, bp_a] = butter(4, BPRange/(fs/2)); 
% 
% for i = 1:size(speechSegs, 1)
%     filtSeg = filtfilt(hp_b, hp_a, sSegTime{i});
%     filtSeg = filtfilt(bp_b, bp_a, filtSeg);
%     sSegTime{i} = filtSeg;
% end
% numSegs = length(sSegTime);
% crossCorrsT = cell(numSegs, numSegs);
% crossCorrsF = cell(numSegs, numSegs);
% corrCoeffT = cell(numSegs, numSegs);
% corrCoeffF = cell(numSegs, numSegs);

%% newlk
% for i = 1:numSegs
%     for j = i+1:numSegs
% 
%         [crossCorrsT{i, j},lagsT]= xcorr(sSegTime{i}, sSegTime{j});
%         [~, maxIdxT] = max(abs(crossCorrsT{i,j})); 
%         bestLagT = lagsT(maxIdxT); 
% 
%         % [crossCorrsF{i,j}, lagsF] = xcorr(sSegFreq{i}, sSegFreq{j});
%         % [~, maxIdxF] = max(abs(crossCorrsF{i,j}));
%         % bestLagF = lagsF(maxIdxF);
%         if bestLagT > 0
%             x1 = sSegTime{i}(bestLagT+1:end); 
%             x2 = sSegTime{j}(1:min(length(sSegTime{j}), length(x1)));
%             x1 = x1(1:length(x2));
%         else
%             x2 = sSegTime{j}(-bestLagT+1:end); 
%             x1 = sSegTime{i}(1:min(length(sSegTime{i}), length(x2)));
%             x2 = x2(1:length(x1));
%         end
%            X1 = fftshift(fft(x1)/length(x1));
%            X2 = fftshift(fft(x2)/length(x2));
        % if bestLagT > 0
        %     X1 = sSegFreq{i}(bestLag+1:end); 
        %     X2 = sSegFreq{j}(1:min(length(sSegFreq{j}), length(X1)));   
        %     X1 = X1(1:length(X2));
        % else
        %     X2 = sSegFreq{j}(-bestLag+1:end); 
        %     X1 = sSegFreq{i}(1:min(length(sSegFreq{i}), length(X2)));
        %     X2 = X2(1:length(X1));
        % end
        % if length(sSegTime{i}) < length(sSegTime{j})
        %     x1 = sSegTime{i};
        %     x2 = sSegTime{j};
        % else
        %     x2 = sSegTime{i};
        %     x1 = sSegTime{j};
        % end

        % if length(sSegFreq{i}) < length(sSegFreq{j})
        %     X1 = sSegFreq{i};
        %     X2 = sSegFreq{j};
        % else
        %     X2 = sSegFreq{i};
        %     X1 = sSegFreq{j};
        % end
        % 
        % x2 = x2(1:length(x1));
        % X2 = X2(1:length(X1));
        % x_short = linspace(1, length(x1), length(x1));
        % x_long = linspace(1, length(x1), length(x2));
        % 
        % X_short = linspace(1, length(X1), length(X1));
        % X_long = linspace(1, length(X1), length(X2));
        % 
        % x1 = interp1(x_short, x1, x_long, 'linear');
        % X1 = interp1(X_short, X1, X_long, 'linear');

        % 
        % corrMatrixT = corrcoef(x1', x2');
        % corrMatrixF = corrcoef(X1', X2');
        % 
        % corrCoeffT{i, j} = corrMatrixT(1,2);
        % corrCoeffF{i, j} = corrMatrixF(1,2);
        % 
        % 
        % % [dist, ix, iy] = dtw(sSegTime{i}, sSegTime{j});
        % % 
        % % fprintf('DTW distance between segment %d and segment %d: %.4f\n', i, j, dist);
        % 
        % fprintf('Correlation between segment %d and segment %d (Time Domain): %.4f\n', i, j, corrCoeffT{i, j});
        % fprintf('Correlation between segment %d and segment %d (Frequency Domain): %.4f\n', i, j, corrCoeffF{i, j});
        % 
        % 
        % figure;
        % subplot(2,1,1);
        % plot((0:length(x1)-1)/fs, x1);
        % xlabel('Time (s)');
        % ylabel('Amplitude');
        % title(['Segment ', num2str(i), ' Time Domain']);
        % 
        % subplot(2,1,2);
        % plot((0:length(x2)-1)/fs, x2);
        % xlabel('Time (s)');
        % ylabel('Amplitude');
        % title(['Segment ', num2str(j), ' Time Domain']);
        % 
        % %% Plot Frequency-Domain Signals
        % 
        % f_plot = (-length(X1)/2:length(X1)/2-1)*fs/length(X1);
        % figure;
        % subplot(2,1,1);
        % plot(f_plot, abs(X1));
        % xlabel('Frequency (Hz)');
        % ylabel('Magnitude');
        % title(['Segment ', num2str(i), ' Frequency Domain']);
        % xlim([-3000, 3000]);
        % 
        % subplot(2,1,2);
        % plot(f_plot, abs(X2));
        % xlabel('Frequency (Hz)');
        % ylabel('Magnitude');
        % title(['Segment ', num2str(j), ' Frequency Domain']);
        % xlim([-3000, 3000]);
%        % 
%        %  % corrCoeffF(i, j) = customWindowedCorrelation(sSegFreq{i}', sSegFreq{j}',.5);
%        %  crossCorrsF{i,j} = xcorr(sSegFreq{i},sSegFreq{j});
%        % [idc, lagF] = xcorr(sSegFreq{i},sSegFreq{j});
%        % 
%        %  figure
%        %  plot(lagT, crossCorrsT{i,j});
%        %  xlabel('lag')
%        %  ylabel('Correlation')
%        %  title(['Cross Correlation (Time Domain) for Segments ',num2str(i),' and ', num2str(j)])
%        % 
%        %  figure
%        %  plot(lagF, crossCorrsF{i,j});
%        %  xlabel('lag')
%        %  ylabel('Correlation')
%        %  title(['Cross Correlation (Freq Domain) for Segments ',num2str(i),' and ', num2str(j)])
% 
%     end
% end
% 
% 
% % %% Finding good SNR thresh for noise
% % avgRealNoisePower = mean(avgPow_N);
% % noiseCoefs = [0.1, 0.2, 0.5, .7, 1, 1.5, 2, 2.5, 3, 4, 5];  
% % for i = 1:length(noiseLevels)
% %     noiseLevels = noiseCoefs(i).*avgRealNoisePower;
% % end
% % snrThresholds = 0:2:20;  
% % bestThreshold = zeros(length(noiseLevels), 1);
% % bestCorr = zeros(length(noiseLevels), 1);
% 
% % for n = 1:length(noiseLevels)
% %     noiseLevel = noiseLevels(n);
% 
% %     noisySTFT = sigSTFT;  
% %     for i = 1:size(sigSTFT, 2) 
% %         whiteNoise = noiseLevels * randn(size(sigSTFT(:, i)));  
% %         noisySTFT(:, i) = sigSTFT(:, i) + whiteNoise;  
% %     end
% 
% %     avgPow_S_noisy = zeros(1, size(sigSTFT, 2));
% %     avgPow_N_noisy = noiseLevel^2 * ones(1, size(sigSTFT, 2)); 
% 
% %     % Calculate the power of each frame in the noisy STFT
% %     for i = 1:size(sigSTFT, 2)
% %         avgPow_S_noisy(i) = mean(avgPow_S);
% %     end
% %     % Calculate SNR in dB for each frame
% %     SNR_noisy = 10 * log10(avgPow_S_noisy ./ avgPow_N_noisy); 
% 
% %     maxAvgCorr = -inf;  % Initialize max correlation
% %     bestSNRThresh = 0;
% 
% %     % Iterate over each threshold
% %     for t = 1:length(snrThresholds)
% %         SNR_thresh = snrThresholds(t);  
% 
% %         % Apply the threshold to the SNR to find speech frames
% %         SNR_filt_noisy = movmedian(SNR_noisy, 9);  % Filter the SNR
% %         speechFrames_noisy = find(SNR_filt_noisy > SNR_thresh);  % Frames exceeding the threshold
% 
% %         speechSegs_noisy = [];
% %         if ~isempty(speechFrames_noisy)
% %             % Identify continuous speech segments based on frames
% %             startIdx = speechFrames_noisy(1);
% %             for i = 2:length(speechFrames_noisy)
% %                 if speechFrames_noisy(i) > speechFrames_noisy(i-1) + 5
% %                     speechSegs_noisy = [speechSegs_noisy; startIdx, speechFrames_noisy(i-1)];
% %                     startIdx = speechFrames_noisy(i); 
% %                 end
% %             end
% %             speechSegs_noisy = [speechSegs_noisy; startIdx, speechFrames_noisy(end)];
% %         end
% 
% %         corrSum = 0;
% %         numCorrs = 0;
% %         % Compute correlation between speech segments in the noisy STFT
% %         for i = 1:size(speechSegs_noisy,1)
% %             for j = i+1:size(speechSegs_noisy,1)
% %                 len1 = length(noisySTFT(:, speechSegs_noisy(i, 1)));
% %                 len2 = length(noisySTFT(:, speechSegs_noisy(j, 1)));
% %                 minLen = min(len1, len2);
% 
% %                 seg1 = abs(noisySTFT(1:minLen, speechSegs_noisy(i, 1)));
% %                 seg2 = abs(noisySTFT(1:minLen, speechSegs_noisy(j, 1)));
% 
% %                 corrMatrixFreq = corrcoef(seg1, seg2);
% 
% %                 corrSum = corrSum + corrMatrixFreq(1, 2); 
% %                 numCorrs = numCorrs + 1;
% %             end
% %         end
% 
% %         avgCorr = corrSum / numCorrs;
% 
% %         if avgCorr > maxAvgCorr
% %             maxAvgCorr = avgCorr;
% %             bestSNRThresh = SNR_thresh;
% %         end
% %     end
% 
% %     bestThreshold(n) = bestSNRThresh;
% %     bestCorr(n) = maxAvgCorr;
% % end
% 
% % disp('Best SNR thresholds for different noise levels:');
% % disp(bestThreshold);
% % disp('Maximum correlation for each noise level:');
% % disp(bestCorr);
% 
% % % Plot results
% % figure;
% % plot(noiseLevels, bestCorr, '-o');
% % xlabel('Noise Level');
% % ylabel('Maximum Correlation');
% % title('Max Correlation vs. Noise Level');
% 
% % figure;
% % plot(noiseLevels, bestThreshold, '-o');
% % xlabel('Noise Level');
% % ylabel('Best SNR Threshold');
% % title('Best SNR Threshold vs. Noise Level');
% 
% 
% % avgNoiseLevelTest = mean(noiseLevels.^2);  % Test noise power (from noiseLevels)
% % avgRealNoisePower = mean(avgPow_N);  % Real ambient noise power (from avgPow_N)
% 
% % % Calculate the ratio between real noise power and test noise power
% % realToTestNoiseRatio = avgRealNoisePower / avgNoiseLevelTest;
% 
% % % Step 3: Scale the best threshold found in the noise test using the ratio
% % scaledSNRThresh = bestThreshold * realToTestNoiseRatio;
% 
% % % Step 4: Apply the adjusted SNR threshold to the real data
% % SNR_filt_real = movmedian(SNR, 9);  % Filter the SNR (from the real signal)
% % speechFrames_real = find(SNR_filt_real > scaledSNRThresh);  % Frames exceeding the adjusted threshold
% 
% % % Identify real speech segments based on the adjusted threshold
% % speechSegs_real = [];
% % if ~isempty(speechFrames_real)
% %     startIdx = speechFrames_real(1);
% %     for i = 2:length(speechFrames_real)
% %         if speechFrames_real(i) > speechFrames_real(i-1) + 5
% %             speechSegs_real = [speechSegs_real; startIdx, speechFrames_real(i-1)];
% %             startIdx = speechFrames_real(i); 
% %         end
% %     end
% %     speechSegs_real = [speechSegs_real; startIdx, speechFrames_real(end)];
% % end
% 
% % % Plot the real speech segments using the adjusted threshold
% % figure;
% % plot(t, SNR_filt_real);
% % hold on;
% % for i = 1:size(speechSegs_real, 1)
% %     xline(speechSegs_real(i, 1), '--r', 'Start Speech');
% %     xline(speechSegs_real(i, 2), '--g', 'End Speech');
% % end
% % xlabel('Time (s)');
% % ylabel('SNR (dB)');
% % title('Filtered SNR with Detected Speech Segments (Real Ambient Noise)');
% hold off;