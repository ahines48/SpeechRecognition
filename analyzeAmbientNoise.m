function powerNoise = analyzeAmbientNoise(device)
    adr = device;

    latency = adr.SamplesPerFrame / adr.SampleRate;
    fs = adr.SampleRate;
    frameRate = 40;
    frameSize = floor(fs/frameRate);
    buffer = dsp.AsyncBuffer(frameSize);

    window = hann(frameSize);
    overlap = floor((floor(fs / frameRate)) / 1.5);
    Ndft = max(256, 2^nextpow2(length(window)));
    
    timeLim = 10;  
    tic
    i = 1;
    t = (0:fs-1) / fs;
    y = zeros(frameSize, timeLim * frameRate);
    totalY = [];
    all_s = [];
    tic
    while toc < timeLim
        [audioIn, overrun(i)] = adr();
        write(buffer, audioIn .* window);
        y(:, i) = read(buffer, frameSize, frameSize - adr.SamplesPerFrame);
        [s, f, t] = stft(y(:,i), fs, Window=window, OverlapLength=overlap, FFTLength=Ndft);
        all_s = [all_s, s];
        i = i + 1;
    end
   
    save('s.mat','s')
    powerNoise = mean(abs(all_s).^2 / length(all_s), 'all');
  
    % Save the result to a .mat file (optional)
    save('ambientNoise.mat', 'powerNoise');
    figure;
    spectrogram(y(:), window, overlap, Ndft, fs, 'yaxis');
    title('STFT Spectrogram of the Last Column of y');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
end
