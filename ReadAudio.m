clear all
close all

fs = 96e3;
frameRate = 9.8;
frameSize = floor(fs/frameRate);
dataPrecision = '24-bit integer'; 
device = 'Sound Blaster G3';
adr = audioDeviceReader('Device',device,'SampleRate',fs ...
    ,'SamplesPerFrame',frameSize,'BitDepth',dataPrecision);
latency = adr.SamplesPerFrame/adr.SampleRate;


buffer = dsp.AsyncBuffer(frameSize);
reset(buffer);
window = hann(frameSize);
overlap = floor((floor(fs/frameRate))/1.5);


h1 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
h2 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

fftFreq = (0:frameSize-1)*fs/frameSize;

stftTotal = [];
timeLim = 10;
timeTotal = [];
powerTotal = [];
elapsedTime = 0;
numSamples = 0;
i = 1;
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
     % Calculate the time increment per frame, considering the overlap
    timeIncrement = (frameSize - overlap) / fs;
    
    % Update elapsed time using actual time passed instead of samples
    elapsedTime = toc;  % Use real elapsed time from the system
    
    % Calculate current time based on elapsed time
    curTime = elapsedTime - (frameSize-overlap)/fs + (0:(frameSize-1))/fs;

    timeTotal = [timeTotal, curTime];

    figure(h1);
    plot(curTime,y(:,i))
    axis tight
    ylim([-.3,.3]) 
    xlabel('time (s)'), ylabel('Amplitude'), title('Audio stream')
    
    % STFT stuff
    signalWindowed = y(:, i).*window;
    fftAudio= fftshift(fft(signalWindowed));
    fftMag = abs(fftAudio);
    powerSpectrum = 20*log10((fftMag.^2)/length(signalWindowed)); 
    stftTotal(:,i) = fftMag;

    figure(h2)
    subplot(2,1,1)
    plot(curTime, signalWindowed);
    axis tight;
    ylim([-.3, .3]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Windowed Audio stream');

    figure(h2) 
    subplot(2,1,2);
    plot(fftFreq, powerSpectrum)
    xlabel('Frequency (Hz)')
    ylabel('Power')
    title('Audio Spectral Density Map')
    xlim([0, 5000])
   

    i = i+1; 
end

figure;
imagesc(timeTotal, fftFreq, 20*log10(abs(stftTotal).^2)/length(stftTotal));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;

release(adr)
release(buffer)