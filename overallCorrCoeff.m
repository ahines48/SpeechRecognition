function overallCorrCoeff = customWindowedCorrelation(seg1, seg2, windowOverlap)

seg1 = seg1(:)';
seg2 = seg2(:)';

len1 = length(seg1);
len2 = length(seg2);
avgLen = floor((len1 + len2) / 2); 
windowSize = avgLen; 
hopSize = floor(windowSize * (1 - windowOverlap)); 

corrs = [];
numWindows = 0; 

for startIdx = 1:hopSize:min(len1, len2) - windowSize + 1
    % Extract windows from both segments
    windowSeg1 = seg1(startIdx:startIdx + windowSize - 1);
    windowSeg2 = seg2(startIdx:startIdx + windowSize - 1);
    
    % Step 4: Normalize each window
    windowSeg1 = (windowSeg1 - mean(windowSeg1)) / std(windowSeg1);
    windowSeg2 = (windowSeg2 - mean(windowSeg2)) / std(windowSeg2);
    
    % Step 5: Compute the correlation coefficient for the window
    corrWindow = corr(windowSeg1', windowSeg2');
    
    % Store the correlation for this window
    corrs = [corrs, corrWindow];
    
    % Increment the number of windows
    numWindows = numWindows + 1;
end

% Step 6: Compute the overall correlation coefficient by averaging the windowed correlations
if numWindows > 0
    overallCorrCoeff = mean(corrs); % Average of all window correlations
else
    overallCorrCoeff = NaN; % If no windows were compared, return NaN
    warning('No windows were processed. Check the segment lengths or window size.');
end

end
