function out = residuals(Y, Yhat, X, opts)
% Inputs: Y is truth, Yhat is model output, X optionally gives alpha/beta locations.
% Outputs: out stores residuals, RMS, distribution stats and optional surface data.
% RESIDUALS Model-error analysis: residual time-series + histogram + (α,β)

    % Defaults & shape checks
    if nargin < 3, X    = []; end
    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'label'), opts.label = ''; end
    if ~isfield(opts, 'nBins'), opts.nBins = 31; end

    Y    = Y(:);
    Yhat = Yhat(:);
    if numel(Y) ~= numel(Yhat)
        error('validation:residuals:sizeMismatch', ...
              'Y has %d elements, Yhat has %d. Must match.', ...
              numel(Y), numel(Yhat));
    end

    % Residuals - full-length, NaNs preserved so caller sees them.
    eps_full = Y - Yhat;

    % For SCALAR STATS we drop NaN samples (out-of-hull predictions etc.).
    mask     = ~isnan(eps_full);
    eps_clean = eps_full(mask);
    N         = numel(eps_clean);

    % Headline statistics
    out.eps  = eps_full;
    out.N    = N;

    if N == 0
        % Degenerate: every prediction was NaN. Return zeros and warn.
        warning('validation:residuals:allNaN', ...
                'All residuals are NaN - no statistics computed.');
        out.RMS      = NaN;
        out.mean     = NaN;
        out.std      = NaN;
        out.skewness = NaN;
        out.kurtosis = NaN;
        out.histEdges  = [];
        out.histCounts = [];
        out.histFitMu    = NaN;
        out.histFitSigma = NaN;
        out.label = opts.label;
        return;
    end

    out.RMS  = sqrt(mean(eps_clean .^ 2));
    out.mean = mean(eps_clean);
    out.std  = std(eps_clean);

    % skewness / kurtosis with flag=1 → raw definitions (no bias correction).
    out.skewness = skewness(eps_clean, 1);
    out.kurtosis = kurtosis(eps_clean, 1) - 3;

    [counts, edges] = histcounts(eps_clean, opts.nBins);
    out.histEdges    = edges;
    out.histCounts   = counts;
    out.histFitMu    = out.mean;
    out.histFitSigma = out.std;

    % Optional residual surface over (α, β)
    if ~isempty(X)
        if size(X,1) ~= numel(eps_full)
            error('validation:residuals:XSizeMismatch', ...
                  'X has %d rows but Y has %d. Must match.', ...
                  size(X,1), numel(eps_full));
        end
        if size(X,2) ~= 2
            error('validation:residuals:XCols', ...
                  'X must be N×2 (columns = α, β). Got %d cols.', size(X,2));
        end
        out.surfaceData.X          = X;
        out.surfaceData.eps        = eps_full;
        out.surfaceData.alphaRange = [min(X(:,1)), max(X(:,1))];
        out.surfaceData.betaRange  = [min(X(:,2)), max(X(:,2))];
    end

    out.label = opts.label;
end
