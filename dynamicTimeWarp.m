function [meanSquareErr, optimalPath] = dynamicTimeWarp(STFT_Data, Dictionary_Data)
    timeWarp = inf([size(STFT_Data,2)+1 size(Dictionary_Data,2)+1]);
    timeWarp(1,1) = 0;
    n = size(timeWarp, 2);
    k = size(timeWarp, 1);
    optimalPath = {};
    meanSquareErr = 0;
    for i=2:size(STFT_Data,2)
        for j=2:size(Dictionary_Data,2)
            cost = sum(abs(STFT_Data(:,i)-Dictionary_Data(:,j)));
            timeWarp(i,j) = cost+min([timeWarp(i-1,j), timeWarp(i, j-1), timeWarp(i-1, j-1)]);
        end

    end

    while n > 1 || k > 1
        if k == 1
            n = n-1;
        elseif n == 1 
            k = k-1;
        else
            [~, minIndex] = min([timeWarp(k-1, n), timeWarp(k, n-1), timeWarp(k-1, n-1)]);
            if minIndex == 1
                k = k-1;
            elseif minIndex == 2
                n = n-1;
            else
                k = k-1;
                n = n-1;
            end
            optimalPath{end+1} = [k n];
        end
    end
    optimalPath = flip(optimalPath);
   for l=1:length(optimalPath)
       coords = optimalPath{l};
       x = coords(1);
       y = coords(2);
       meanSquareErr = meanSquareErr + timeWarp(x,y);
       meanSquareErr = meanSquareErr/length(optimalPath);
   end
end