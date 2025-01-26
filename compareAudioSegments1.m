
signal = load('Dog_signal.mat');
signal = signal.sSegTime;
compareAudioSegments(signal{1})


function [word, corrCoeff] = compareAudioSegments1(SegFreq, noise)
    fs = 96e3;
    dog = load('Dog_CleanRun3.mat');
    cat = load('Cat_CleanRun4.mat');
    bird = load('Bird_CleanRun3.mat');
    fish = load('Fish_CleanRun3.mat');
    mouse = load('Mouse_CleanRun3.mat');
    avgNoise = mean(noise(:));

    SegFreq = SegFreq./(SegFreq+avgNoise); 
    
    dog = dog.sSegFreq;
    cat = cat.sSegFreq;
    bird = bird.sSegFreq;
    fish = fish.sSegFreq;
    mouse = mouse.sSegFreq;
    dict = {dog, cat, bird, fish, mouse};
    crossCorrs = cell(size(dict));
    % SegTime = (SegTime-mean(SegTime))/std(SegTime);
    corrCoeff = zeros([5, 5]);
    for i = 1:5
        cur_word = dict{i};
        % for j = 1
            dictSignal = cur_word{1};
            dictSignal = reshape(dictSignal, 9600, length(dictSignal/9600));
            % dictSignal = (dictSignal-mean(dictSignal))/std(dictSignal);
       
            
            [crossCorr, lags] = xcorr(SegTime, dictSignal);
            [~, maxIdx] = max(abs(crossCorr));
            bestLag = lags(maxIdx);
            
            
            if bestLag > 0
                X1 = SegTime(bestLag+1:end);
                X2 = dictSignal(1:min(length(dictSignal), length(X1)));
                X1 = X1(1:length(X2));
            else
                X2 = SegTime(-bestLag+1:end);
                X1 = SegTime(1:min(length(SegTime), length(X2)));
                X2 = X2(1:length(X1));
            end
            X1 = fftshift(fft(X1)/length(X1));
            X2 = fftshift(fft(X2)/length(X2));

            corrMatrix= corrcoef(X1', X2');
            corrCoeff(i,1) = corrMatrix(1, 2);
            fprintf('Correlation between segment signal and segment %d (Frequency Domain): %.4f\n', i, corrCoeff(i,1));
            if corrMatrix(1,2) >.85
                break;
            end
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
    % title(['Segment ', num2str(i), ' Time Domain']);


    f_plot = (-length(X1)/2:length(X1)/2-1)*fs/length(X1);
    figure;
    subplot(2,1,1);
    plot(f_plot, abs(X1));
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title(['Segment ', num2str(i), ' Frequency Domain']);
    xlim([-3000, 3000]);

    subplot(2,1,2);
    plot(f_plot, abs(X2));
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title(['Segment ', num2str(j), ' Frequency Domain']);
    xlim([-3000, 3000]);

        % end
    end
    [maxCorr, linearIdx] = max(corrCoeff(:));
    [maxRow, ~] = ind2sub(size(corrCoeff), linearIdx);

    switch maxRow
        case 1
            word = 'dog';
        case 2
            word = 'cat';
        case 3
            word = 'bird';
        case 4
            word = 'fish';
        case 5
            word = 'mouse';
        otherwise
            word = 'unknown';
    end

    fprintf('The word being said is: %s\n', word);
end