mouse = load('mouse_1_1620_Lab.mat');
mouse = mouse.sSegFreq;
mouse = mouse{1};

Fs = 16000;
time = linspace(0, size(mouse, 2), size(mouse, 2)); 
freq = linspace(0, Fs/2, size(mouse, 1)); 

figure;
imagesc(time, freq, 20*log10(mouse));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram (dB)');
colorbar;