clear all
close all

device = 'Sound Blaster G3';
fs = 96e3;
frameRate = 25; %5 plot 25 no plot
timeLim1 = 3;
timeLim2 = 10;
frameSize = floor(fs/frameRate);


[noiseRaw, noiseSTFT, noisePow, noiseTime, noiseFreq] = audio_analysis_function(fs, frameRate, device, timeLim1);
[sigRaw, sigSTFT, sigPow, sigTime, sigFreq] = audio_analysis_function(fs, frameRate, device, timeLim2);


%% Identify Speech Segments
overlap = 1/2;
hopSize = frameRate * (1 - overlap);
frameTime = (frameSize+hopSize) / fs;
colSig = size(sigPow, 2);
colNoise = size(noisePow, 2);
repeatFactor = ceil(colSig / colNoise); 
noisePowRepeated = repmat(noisePow, 2, repeatFactor);
noisePow = noisePowRepeated(:, 1:colSig);
t = (0:colSig-1) * frameTime;

avgPow_S = zeros(1,colSig);
avgPow_N = zeros(1,colSig);

for i = 1:colSig
    avgPow_S(i) = sum(sigPow(:,i))/size(sigPow,1);
    avgPow_N(i) = sum(noisePow(:,i))/size(noisePow,1);
end
SNR = 20*log10(avgPow_S./avgPow_N); 
plotFrames = 1:length(avgPow_S);
SNR_filt = movmedian(SNR, 9);
SNR_thresh = 10;
speechFrames = find(SNR_filt > SNR_thresh);
speechSegs = [];

if ~isempty(speechFrames)
    startIdx = speechFrames(1);
    for i = 2:length(speechFrames)
        if speechFrames(i) > speechFrames(i-1) + 5
            speechSegs = [speechSegs; startIdx, speechFrames(i-1)];
            startIdx = speechFrames(i); 
        end
        
    end
    speechSegs = [speechSegs; startIdx, speechFrames(end)];
end

speechTimes = zeros(size(speechSegs));

for i = 1:size(speechSegs, 1)
    startFrame = speechSegs(i, 1);
    endFrame = speechSegs(i, 2);
    
    startTime = (startFrame - 1) * frameTime;  
    endTime = (endFrame - 1) * frameTime;  
    
    speechTimes(i, :) = [startTime, endTime];
end

plotSigRaw = reshape(sigRaw, [], 1);
t_plot =  (0:length(plotSigRaw)-1) / fs;
plotSigSTFT = reshape(sigSTFT, [], 1);
f_plot = (0:length(plotSigSTFT)-1)*fs/length(plotSigSTFT);


figure;
plot(t_plot, plotSigRaw); 
xlabel('Time (s)');
ylabel('Amplitude');
title('Full Audio Signal vs. Time');
hold on;

for i = 1:size(speechTimes, 1)
    xline(speechTimes(i, 1), '--r', 'Start Speech');
    xline(speechTimes(i, 2), '--g', 'End Speech');
end

hold off;

figure;
imagesc(sigTime, sigFreq, 20*log10(abs(sigSTFT).^2));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram w/ Speech Segments');
colorbar;
hold on;
for i = 1:size(speechTimes, 1)
    xline(speechTimes(i, 1), '--r', 'Start Speech');
    xline(speechTimes(i, 2), '--g', 'End Speech');
end
hold off;

figure
plot(plotFrames, SNR);
xlabel('Frame Num')
ylabel('SNR')
title('SNR vs Frame Number')

figure
plot(t, SNR);
xlabel('Time (s)')
ylabel('SNR (dB)')
title('SNR vs Time')

figure;
plot(t, SNR_filt);
hold on;
for i = 1:size(speechTimes, 1)
    xline(speechTimes(i, 1), '--r', 'Start Speech');
    xline(speechTimes(i, 2), '--g', 'End Speech');
end
xlabel('Time (s)');
ylabel('SNR (dB)');
title('Filtered SNR with Detected Speech Segments');
hold off;

figure;
plot(t, SNR);
hold on;
for i = 1:size(speechTimes, 1)
    xline(speechTimes(i, 1), '--r', 'Start Speech');
    xline(speechTimes(i, 2), '--g', 'End Speech');
end
xlabel('Time (s)');
ylabel('SNR (dB)');
title('SNR with Detected Speech Segments');
hold off;


%% Correlation Between Segments
sSegTime = {};
sSegFreq = {};
for i = 1:size(speechSegs, 1)
    startFrame = speechSegs(i, 1);
    endFrame = speechSegs(i, 2);

    sigMatrixT = sigRaw(:,startFrame:endFrame);
    sigVecT = reshape(sigMatrixT,1,[]);
    sSegTime{i} = sigVecT;
    
    sigMatrixF = sigSTFT(:,startFrame:endFrame);
    sigVecF = reshape(sigMatrixF,1,[]);
    sSegFreq{i} = sigVecF;

end
HPCutoff = 500;
BPRange = [150 4000];
[hp_b, hp_a] = butter(4, HPCutoff/(fs/2), 'high');
[bp_b, bp_a] = butter(4, BPRange/(fs/2)); 

for i = 1:size(speechSegs, 1)
    filtSeg = filtfilt(hp_b, hp_a, sSegTime{i});
    filtSeg = filtfilt(bp_b, bp_a, filtSeg);
    sSegTime{i} = filtSeg;
end
numSegs = length(sSegTime);
crossCorrsT = cell(numSegs, numSegs);
crossCorrsF = cell(numSegs, numSegs);
corrCoeffT = cell(numSegs, numSegs);
corrCoeffF = cell(numSegs, numSegs);

for i = 1:numSegs
    for j = i+1:numSegs

        [crossCorrsT{i, j},lagsT]= xcorr(sSegTime{i}, sSegTime{j});
        [~, maxIdxT] = max(abs(crossCorrsT{i,j})); 
        bestLagT = lagsT(maxIdxT); 
        
        [crossCorrsF{i,j}, lagsF] = xcorr(sSegFreq{i}, sSegFreq{j});
        [~, maxIdxF] = max(abs(crossCorrsF{i,j}));
        bestLagF = lagsF(maxIdxF);
        if bestLagT > 0
            x1 = sSegTime{i}(bestLagT+1:end); 
            x2 = sSegTime{j}(1:min(length(sSegTime{j}), length(x1)));
            x1 = x1(1:length(x2));
        else
            x2 = sSegTime{j}(-bestLagT+1:end); 
            x1 = sSegTime{i}(1:min(length(sSegTime{i}), length(x2)));
            x2 = x2(1:length(x1));
        end

        if bestLagF > 0
            X1 = sSegFreq{i}(bestLagF+1:end); 
            X2 = sSegFreq{j}(1:min(length(sSegFreq{j}), length(X1)));   
            X1 = X1(1:length(X2));
        else
            X2 = sSegFreq{j}(-bestLagF+1:end); 
            X1 = sSegFreq{i}(1:min(length(sSegFreq{i}), length(X2)));
            X2 = X2(1:length(X1));
        end
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

       
        corrMatrixT = corrcoef(x1', x2');
        corrMatrixF = corrcoef(X1', X2');

        corrCoeffT{i, j} = corrMatrixT(1,2);
        corrCoeffF{i, j} = corrMatrixF(1,2);


        % [dist, ix, iy] = dtw(sSegTime{i}, sSegTime{j});
        % 
        % fprintf('DTW distance between segment %d and segment %d: %.4f\n', i, j, dist);

        fprintf('Correlation between segment %d and segment %d (Time Domain): %.4f\n', i, j, corrCoeffT{i, j});
        fprintf('Correlation between segment %d and segment %d (Frequency Domain): %.4f\n', i, j, corrCoeffF{i, j});

        
        figure;
        subplot(2,1,1);
        plot((0:length(x1)-1)/fs, x1);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(['Segment ', num2str(i), ' Time Domain']);

        subplot(2,1,2);
        plot((0:length(x2)-1)/fs, x2);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(['Segment ', num2str(j), ' Time Domain']);

        %% Plot Frequency-Domain Signals
        figure;
        subplot(2,1,1);
        plot((0:length(X1)-1)*fs/length(X1), abs(X1));
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        title(['Segment ', num2str(i), ' Frequency Domain']);

        subplot(2,1,2);
        plot((0:length(X2)-1)*fs/length(X2), abs(X2));
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        title(['Segment ', num2str(j), ' Frequency Domain']);
        
        %corrCoeffF(i, j) = customWindowedCorrelation(sSegFreq{i}', sSegFreq{j}',.5);
        %crossCorrsF{i,j} = xcorr(sSegFreq{i},sSegFreq{j});
       % [idc, lagF] = xcorr(sSegFreq{i},sSegFreq{j});

        % figure
        % plot(lagT, crossCorrsT{i,j});
        % xlabel('lag')
        % ylabel('Correlation')
        % title(['Cross Correlation (Time Domain) for Segments ',num2str(i),' and ', num2str(j)])
        % 
        % figure
        % plot(lagF, crossCorrsF{i,j});
        % xlabel('lag')
        % ylabel('Correlation')
        % title(['Cross Correlation (Freq Domain) for Segments ',num2str(i),' and ', num2str(j)])

    end
end



