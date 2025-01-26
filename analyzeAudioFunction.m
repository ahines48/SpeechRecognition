function [audioData, stftData, powerData, timeData, freqData, fftData, speechT, speechF, correlations, wordGuesses, detectedNegative, SNR_filt] = analyzeAudioFunction(fs, frameRate, device, timeLim, noisePower, noiseSTFT)
    frameSize = floor(fs/frameRate);
    dataPrecision = '24-bit integer'; 
    adr = audioDeviceReader('Device', device, 'SampleRate', fs, ...
        'SamplesPerFrame', frameSize, 'BitDepth', dataPrecision);
    latency = adr.SamplesPerFrame/adr.SampleRate;

    spokenWordResults = {};
    correlations = {};
    buffer = dsp.AsyncBuffer(frameSize);
    reset(buffer);
    overlap = floor(frameSize/2);
    hopSize = frameSize-overlap;
    frameTime = (frameSize+hopSize) / fs;
    h1 = figure;

    figure(h1)
    subplot(2, 1, 1); 
    rawAudioPlot = plot(nan, nan);
    hMarkers = line(nan, nan);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Raw Audio Data');
    xlim([0 timeLim]);

    
    subplot(2, 1, 2); 
    spectrogramPlot = imagesc(zeros(1, 1), zeros(1, 1), zeros(1, 1)); 
    hMarkers1 = line(nan, nan);
    axis xy;
    xlabel('Time (s)');
    set(gca, 'YDir', 'normal');
    ylabel('Frequency (Hz)');
    title('STFT Spectrogram');
    colorbar;
    xlim([0 timeLim]);
    ylim([-6e3 6e3])
    clim([-200 -40])

    lowCutoff = 60;   
    highCutoff = 6000; 
    queue = {};
    
    Wn = [lowCutoff, highCutoff] / (fs/2);
    filterOrder = 4;
    [b, a] = butter(filterOrder, Wn, 'bandpass');
    
    maxIterations = ceil(timeLim*frameRate);
    stftTotal = zeros(frameSize+overlap, maxIterations);
    fftTotal = zeros(frameSize+overlap, maxIterations);

    powerTotal = zeros(frameSize+overlap, maxIterations);
    mfccTotal = zeros(13, maxIterations);
    timeTotal = [];
    y = zeros(frameSize, maxIterations);
    overrun = zeros(1, maxIterations);
    corrCoeffs = [];
    elapsedTime=0;
    i=1;
    
    detectedNegative = 0;
    count = 1;
    prevLength = 0;
    colSig = maxIterations;
    colNoise = size(noisePower, 2);
    repeatFactor = ceil(colSig / colNoise); 
    noisePowRepeated = repmat(noisePower, 2, repeatFactor);
    noisePower = noisePowRepeated(:, 1:colSig);
    avgPow_S = zeros(1, colSig);
    avgPow_N = zeros(1, colSig);
    SNR = zeros(1, colSig);
    SNR_filt = zeros(1, colSig);
    SNR_thresh = 20;
    speechFrames = [];
    SpeechSegsF = {};
    spokenwordFuture = [];
    spokenwordResult = '';
    SpeechSegsT = {};
    inWord = false;
    tic;

while toc < timeLim
    [audioIn, overrun(i)] = adr();
    write(buffer,audioIn);
    y(:, i) = read(buffer, frameSize);

    if overrun(i) > 0
        disp('Overrun detected. Skipping this frame.');
        continue; 
    end


    elapsedTime = toc;
    curTime = elapsedTime - (0:(frameSize-1)) / fs;
    timeTotal = [timeTotal, elapsedTime];

    
    set(rawAudioPlot, 'XData', [get(rawAudioPlot, 'XData'), curTime], ...
        'YData', [get(rawAudioPlot,'YData'), y(:, i)']);

    if i == 1
        sigSTFT = filtfilt(b, a,[zeros(overlap, 1); y(:, i)]);     
    else
        sigSTFT = filtfilt(b, a,[y(end-overlap+1:end, i-1); y(1:end, i)]);
    end
    curWindow = hann(length(sigSTFT));        
    signalWindowed = sigSTFT .* curWindow;
    numPoints = length(signalWindowed);
    fftAudio = fftshift(fft(signalWindowed)/numPoints);
    fftMag = abs(fftAudio);
    powerSpectrum = fftMag(:).^2;
    newSTFT = 20*log10(fftMag(:));
    fftFreq = (-numPoints/2:numPoints/2-1) * fs / numPoints;
    newFreq = fftFreq(:);
    stftTotal(:, i) = fftMag;
    fftTotal(:,i) = fftAudio;
    powerTotal(:, i) = powerSpectrum;
    mfccCurrent = mfcc(y(:, i), fs, 'LogEnergy', 'replace');
    mfccTotal(:, i) = mfccCurrent(1,:);
    set(spectrogramPlot, 'XData', [get(spectrogramPlot,'XData'), curTime], 'YData', newFreq);
     
    existingCData = get(spectrogramPlot, 'CData');
    if existingCData == 0
        set(spectrogramPlot, 'CData', newSTFT);
    else
        if size(newSTFT, 1) == size(existingCData, 1)
            set(spectrogramPlot, 'CData', [existingCData newSTFT]);
        end
    end
    
    if mod(i,10) == 0
        drawnow;
    end
   
    
    avgPow_S(i) = mean(powerSpectrum);
    avgPow_N(i) = mean(noisePower(:, i));
    if avgPow_N(i) ~= 0
        SNR(i) = 20*log10(avgPow_S(i)/avgPow_N(i));
    else
        SNR(i) = 0;
    end

    if i >= 10
        SNR_filt = movmedian(SNR, 9);
        
        if SNR_filt(i-7) > SNR_thresh && inWord == false
            inWord = true;
            speechFrames = [speechFrames, i-9];
            set(hMarkers, 'XData', timeTotal(speechFrames), 'YData', y(speechFrames));
            line([timeTotal(i-9) timeTotal(i-9)], ylim, 'Color', 'g', 'LineWidth', 1);
            
            set(hMarkers1, 'XData', timeTotal(speechFrames), 'YData', y(speechFrames));
            line([timeTotal(i-9) timeTotal(i-9)], ylim, 'Color', 'g', 'LineWidth', 1);
            
        end

        if SNR_filt(i-7) < SNR_thresh && inWord == true
            if ~isempty(speechFrames)
                if speechFrames(end) < i-13
                    speechFrames = [speechFrames, i-8];
                    inWord = false;
                    set(hMarkers, 'XData', timeTotal(speechFrames), 'YData', y(speechFrames));
                    line([timeTotal(i-8) timeTotal(i-8)], ylim, 'Color', 'r', 'LineWidth', 1);
        
                    set(hMarkers1, 'XData', timeTotal(speechFrames), 'YData', y(speechFrames));
                    line([timeTotal(i-8) timeTotal(i-8)], ylim, 'Color', 'r', 'LineWidth', 1);
                else
                    speechFrames(end) = [];
                    detectedNegative = detectedNegative+1;
                    inWord = false;
                end
            end
        end

    end
   
    curLength = length(speechFrames);
    if ~isempty(speechFrames) && prevLength+1 < curLength
        startFrame = speechFrames(curLength-1);
        endFrame = speechFrames(curLength);
        sigMatrix = stftTotal(:,startFrame:endFrame);
        sigMatrixT = mfccTotal(:, startFrame:endFrame);
        sigVecT = reshape(sigMatrixT,1,[]);
        sigVecF = reshape(sigMatrix,1,[]);
        
        SpeechSegsT{count} = sigMatrixT;
        SpeechSegsF{count} = sigMatrix;
        % word = compareAudioSegments(sigMatrix, noiseSTFT);
        % fprintf('The word being said is: %s\n', spokenwordResult);
        % spokenWordResults{end+1} = spokenwordResult;
        % correlations{end+1} = corrCoeffs;
        if isempty(spokenwordFuture) || strcmp(spokenwordFuture.State, 'finished')
            if ~isempty(spokenwordFuture) && strcmp(spokenwordFuture.State, 'finished')
                [spokenwordResult, corrCoeffs] = fetchOutputs(spokenwordFuture);
                fprintf('The word being said is: %s\n', spokenwordResult);
                spokenWordResults{end+1} = spokenwordResult;
                correlations{end+1} = corrCoeffs;

            end
            spokenwordFuture = parfeval(@compareAudioSegmentsCorr, 2, sigMatrix, noiseSTFT);

        else
            queue{end+1} = sigMatrix;
        end

        prevLength = curLength;
        count = count+1;
    end
    i = i+1;

end
release(adr);
release(buffer);
while ~isempty(queue)
    if ~isempty(spokenwordFuture) && strcmp(spokenwordFuture.State, 'running')
         wait(spokenwordFuture);  
         [spokenwordResult, corrCoeffs] = fetchOutputs(spokenwordFuture);

         fprintf('The word being said is: %s\n', spokenwordResult);
         spokenWordResults{end+1} = spokenwordResult;
         correlations{end+1} = corrCoeffs;
    end
        sigMatrix = queue{1};
        queue(1) = []; 
        spokenwordFuture = parfeval(@compareAudioSegmentsCorr, 2, sigMatrix, noiseSTFT);
end
figure;
imagesc(timeTotal(:), fftFreq, 20 * log10(stftTotal));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;
xlim([0 timeLim]);
drawnow


t = (0:colSig-1) * frameTime;
figure;
plot(t,SNR)
hold on
yline(SNR_thresh, '--r', 'SNR thresh');
hold off
title('SNR')
xlabel('Time')
ylabel('SNR')
drawnow

figure;
plot(t,SNR_filt)
hold on
yline(SNR_thresh, '--r', 'SNR thresh');
hold off
title('SNR filt')
xlabel('Time')
ylabel('SNR ish')
drawnow

speechF = SpeechSegsF;
speechT = SpeechSegsT;
fftData = fftTotal;
audioData = y;
stftData = stftTotal;
timeData = timeTotal;
freqData = fftFreq;
powerData = powerTotal;

if ~isempty(spokenwordFuture)
    wait(spokenwordFuture);
    [spokenwordResult,corrCoeffs] = fetchOutputs(spokenwordFuture);
    fprintf('The word being said is: %s\n', spokenwordResult);
    spokenWordResults{end+1} = spokenwordResult;
    correlations{end+1} = corrCoeffs;
end
if ~isempty(spokenwordFuture)
    wait(spokenwordFuture);
    [spokenwordResult,corrCoeffs] = fetchOutputs(spokenwordFuture);
    fprintf('The word being said is: %s\n', spokenwordResult);
    spokenWordResults{end+1} = spokenwordResult;
    correlations{end+1} = corrCoeffs;
end

wordGuesses = spokenWordResults;

end