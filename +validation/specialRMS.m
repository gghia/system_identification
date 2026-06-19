function out = specialRMS(predFun, alpha_val, beta_val, Cm_val, hullData)
% Inputs: predFun predicts Cm, validation vectors define the grid, hullData is training support.
% Outputs: out stores RMS values for all, in-hull and out-of-hull validation points.
% SPECIALRMS Final single-number score on the 100-point special-validation

    % Shape checks
    alpha_val = alpha_val(:);
    beta_val  = beta_val(:);
    Cm_val    = Cm_val(:);
    M         = numel(alpha_val);
    if numel(beta_val) ~= M || numel(Cm_val) ~= M
        error('validation:specialRMS:sizeMismatch', ...
              'alpha_val/beta_val/Cm_val must have the same length; got %d, %d, %d.', ...
              M, numel(beta_val), numel(Cm_val));
    end
    if size(hullData, 2) ~= 2
        error('validation:specialRMS:hullCols', ...
              'hullData must be N×2; got %d cols.', size(hullData, 2));
    end
    if ~isa(predFun, 'function_handle')
        error('validation:specialRMS:badPredFun', ...
              'predFun must be a function handle Yhat = predFun(X).');
    end

    % Convex hull of the training data
    K       = convhull(hullData(:,1), hullData(:,2));
    hullX   = hullData(K, 1);
    hullY   = hullData(K, 2);

    % Membership test
    inMask  = inpolygon(alpha_val, beta_val, hullX, hullY);
    outMask = ~inMask;
    nIn     = sum(inMask);
    nOut    = sum(outMask);

    % Predictions
    X    = [alpha_val, beta_val];
    Yhat = predFun(X);
    Yhat = Yhat(:);
    if numel(Yhat) ~= M
        error('validation:specialRMS:predLength', ...
              'predFun returned %d outputs for %d inputs. Must match.', ...
              numel(Yhat), M);
    end

    % Residuals (NaNs preserved in eps)
    eps_all = Cm_val - Yhat;

    % Helper: RMS over a mask, dropping NaNs.
    rmsOver = @(mask) localRMS(eps_all(mask));

    out.RMS_all = rmsOver(true(M,1));
    out.RMS_in  = rmsOver(inMask);
    out.RMS_out = rmsOver(outMask);

    % Diagnostics on Yhat finiteness
    out.Yhat_finite_in  = any(isfinite(Yhat(inMask)));
    out.Yhat_finite_out = any(isfinite(Yhat(outMask)));

    % Pack
    out.nIn        = nIn;
    out.nOut       = nOut;
    out.inHullMask = inMask;
    out.Yhat       = Yhat;

    % One-line summary for the report.
    out.summary = sprintf( ...
        ['Special-validation RMS: all=%.3g (n=%d), in-hull=%.3g (n=%d), ' ...
         'out-of-hull=%.3g (n=%d)'], ...
        out.RMS_all, M, out.RMS_in, nIn, out.RMS_out, nOut);
end

function r = localRMS(eps_vec)
% Inputs: eps_vec is a residual subset, possibly with NaNs.
% Outputs: r is RMS after dropping NaNs, or NaN if no usable values remain.
% Returns NaN (not 0) when the subset is empty or all-NaN, so an empty
    eps_vec = eps_vec(~isnan(eps_vec));
    if isempty(eps_vec)
        r = NaN;
    else
        r = sqrt( mean(eps_vec .^ 2) );
    end
end
