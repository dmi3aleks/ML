function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    tp = 0;
    fp = 0;
    fn = 0;
    predictions = (pval < epsilon);
    for i = 1:size(yval,1)
      p = predictions(i);
      y = yval(i);
      if p != y
        if p == 1
          fp += 1;
        else
          fn += 1;
        end
      else
        if p == 1
          tp += 1;
        end
      end
    end

    prec = tp/(tp + fp);
    rec = tp/(tp + fn);
    F1 = 2 * prec * rec/(prec + rec);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
