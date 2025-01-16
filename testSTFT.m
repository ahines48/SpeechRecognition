device = 'Sound Blaster G3';
fs = 96e3;
frameRate = 5; %5 plot 25 no plot
timeLim = 10;
frameSize = floor(fs/frameRate);

audio_analysis_sinewave(fs, frameRate, timeLim, 500);