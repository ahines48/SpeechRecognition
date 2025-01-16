close all
clear all

fs = 96e3;
frameRate = 40;
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
Ndft = max(256,2^nextpow2(frameSize));

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
h1 = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
hSpec = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

timeLim = 15;
powerNoise = analyzeAmbientNoise(adr);


i = 1;
t = (0:fs-1)/fs;

y = zeros(frameSize, timeLim * frameRate);
idkAtThisPoint = [];
noiseFrames = floor(10 * frameRate);
speechFrames = timeLim * frameRate - noiseFrames;
noisePower = zeros(1, noiseFrames);
speechPower = zeros(1, speechFrames);
noiseIndex = 1;
speechIndex = 1;
SNR_threshold = 25;
SNR_values = []; 
speechSegmentsFFT = {};

tic
while toc < timeLim
    [audioIn, overrun(i)] = adr();
    write(buffer,audioIn.*window)

    y(:,i) = read(buffer,frameSize,overlap);

    figure(h);
    subplot(2,1,1)
    plot(t,y(:,i))
    axis tight
    ylim([-.3,.3]) 
    xlabel('t (s)'), ylabel('x(t)'), title('Audio stream')
    c = mean(powerNoise);
    framePower = sum(y(:, i).^2) / length(y(:, i));  
    SNR = 20 * log10(framePower / mean(powerNoise)); 
    SNR_values = [SNR_values, SNR];

    if SNR > SNR_threshold
        fprintf('Frame %d: Speech Detected. SNR = %.2f dB\n', i, SNR);
        speechPower(speechIndex) = framePower;
        speechIndex = speechIndex + 1;
        fftSAudio = abs(fft(y(:, i)));
        fftSMag = abs(fftSAudio);
        fftSFreq = (0:length(fftSAudio)-1) * fs / length(fftSAudio);
        powerSSpectrum = 20 * log10((fftSMag.^2) / length(y));
        
        speechSegmentsFFT{end+1} = powerSpectrum(1:floor(end/2));
    else
        fprintf('Frame %d: Noise Detected. SNR = %.2f dB\n', i, SNR);
        noisePower(noiseIndex) = framePower;
        noiseIndex = noiseIndex + 1;
    end

    fftAudio= abs(fft(y(:,i)));
    fftMag = abs(fftAudio);
    fftFreq = (0:length(fftAudio)-1) * fs / length(fftAudio);
    powerSpectrum = 20*log10((fftMag.^2) / length(y));
    
    figure(h1) 
    subplot(2,1,2);
    plot(fftFreq(1:floor(end/2)), powerSpectrum(1:floor(end/2)))
    xlabel('Frequency (Hz)')
    ylabel('Power')
    title('Audio Spectral Density Map')
    
    [st,ft,tt] = stft(y(:,i),fs,Window=window,OverlapLength=overlap, FFTLength=Ndft);

    figure(hSpec);
    stft(y(:,i),fs,Window=window,OverlapLength=overlap, FFTLength=Ndft);
    title('Spectrogram of Audio Stream')
    colorbar
    
    idkAtThisPoint = [idkAtThisPoint, y(:,i)];
    i = i+1; 
    
end

figure;
hold on;
for j = 1:length(speechSegmentsFFT)
    plot(fftFreq(1:floor(end/2)), speechSegmentsFFT{j});
end
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Frequency Spectrum of Detected Speech Segments');
hold off;

release(adr)
release(buffer)
%% Further analysis
figure;
plot(10:length(SNR_values), SNR_values(10:end));
xlabel('Frame Index');
ylabel('SNR (dB)');
title('SNR vs. Frame Index');

speechSegments = y(:, SNR_values > SNR_threshold); 
if length(speechSegments) > 1
    figure;
    hold on;
    for j = 1:length(speechSegments)-1
        [crossCorr, lag] = xcorr(speechSegments(:, j), speechSegments(:, j+1));
        plot(lag, crossCorr);
    end
    xlabel('Lag');
    ylabel('Cross-Correlation');
    title('Cross-Correlation of Speech Segments');
    hold off;
end