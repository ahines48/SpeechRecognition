function [audioData,stftData,powerData,timeData,freqData] = audio_analysis_function(fs, frameRate, device, timeLim)
    % Initialize parameters
    frameSize = floor(fs/frameRate);
    dataPrecision = '24-bit integer'; 
    adr = audioDeviceReader('Device',device,'SampleRate',fs ...
        ,'SamplesPerFrame',frameSize,'BitDepth',dataPrecision);
    latency = adr.SamplesPerFrame/adr.SampleRate;

    buffer = dsp.AsyncBuffer(frameSize);
    reset(buffer);
    overlap = floor(frameSize/2);
    hopSize = frameSize - overlap;

    h1 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
    h2 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
    h3 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

    fftFreq = (0:frameSize-1)*fs/frameSize;

    stftTotal = [];
    timeTotal = [];
    powerTotal = [];
    elapsedTime = 0;
    i = 1;
    lowCutoff = 200;   
    highCutoff = 4000; 
    queue = {};
    
    Wn = [lowCutoff, highCutoff] / (fs/2);
    filterOrder = 4;
    [b, a] = butter(filterOrder, Wn, 'bandpass');
    tic;

    while toc < timeLim
        
        % Read Audio
        [audioIn, overrun(i)] = adr();
        write(buffer,audioIn);
        y(:,i) = read(buffer, frameSize);

        if overrun(i) > 0
            disp('Overrun detected. Skipping this frame.');
            continue; 
        end
       
        timeIncrement = (frameSize-overlap)/fs;
        elapsedTime = toc;
        curTime = elapsedTime - (0:(frameSize-1))/fs;
        timeTotal = [timeTotal, elapsedTime];

        figure(h1);
        plot(curTime,y(:,i))
        axis tight
        ylim([-.3,.3]) 
        xlabel('time (s)'), ylabel('Amplitude'), title('Audio stream')

        
       
         if i == 1
            sigSTFT = filtfilt(b, a,[zeros(overlap, 1); y(:, i)]);     
        else
            sigSTFT = filtfilt(b, a,[y(end-overlap+1:end, i-1); y(1:end, i)]);
        end
        curWindow = hann(length(sigSTFT));        
        signalWindowed =sigSTFT.*curWindow;
        numPoints = length(signalWindowed);
        fftAudio= fftshift(fft(signalWindowed)/numPoints);
        fftMag = abs(fftAudio);
        powerSpectrum = fftMag(:).^2;
        fftFreq = (-numPoints/2:numPoints/2-1) * fs / numPoints;


       if isempty(stftTotal) || size(stftTotal, 1) ~= length(fftMag)
            stftTotal = zeros(length(fftMag), 1);
            powerTotal = zeros(length(powerSpectrum), 1);
       end
        
        
        stftTotal(:, i) = fftMag;
        powerTotal(:, i) = powerSpectrum;

        % figure(h2)
        % subplot(3,1,1)
        % plot(curTime, signalWindowed(1:frameSize));
        % axis tight;
        % ylim([-.3, .3]);
        % xlabel('Time (s)');
        % ylabel('Amplitude');
        % title('Windowed Audio stream');
        % 
        % figure(h2) 
        % subplot(3,1,2);
        % plot(fftFreq, powerSpectrum)
        % xlabel('Frequency (Hz)')
        % ylabel('Power')
        % title('Audio Spectral Density Map')
        % xlim([0, 5000])
        % 
        % figure(h2);
        % subplot(3,1,3);
        % imagesc(timeTotal, fftFreq, 20*log10(abs(stftTotal).^2));
        % axis xy;
        % xlabel('Time (s)');
        % ylabel('Frequency (Hz)');
        % title('STFT Spectrogram');
        % colorbar;

        i = i+1; 
    end

    release(adr)
    release(buffer)
    figure(h3);
    
    imagesc(timeTotal, fftFreq, 20*log10(stftTotal));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('STFT Spectrogram');
    colorbar;

    % Ouputs
    audioData = y;
    stftData = stftTotal;
    timeData = timeTotal;
    freqData = fftFreq;
    powerData = powerTotal;
end
