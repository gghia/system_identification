function out = autocorr(eps, opts)
% Inputs: eps is the residual vector, opts can set label and maxLag.
% Outputs: out stores lags, autocorrelation values, confidence band and score.
% AUTOCORR Residual autocorrelation function + 95% white-noise confidence

    if nargin < 2, opts = struct(); end
    if ~isfield(opts, 'label'),  opts.label  = ''; end

    eps = eps(:);

    % Drop NaNs before computing statistics.
    eps = eps(~isnan(eps));
    N   = numel(eps);

    if N < 4
        error('validation:autocorr:tooFewSamples', ...
              'Need at least 4 non-NaN residuals to estimate R̂(τ); got %d.', N);
    end

    if ~isfield(opts, 'maxLag') || isempty(opts.maxLag)
        opts.maxLag = min(50, floor(N / 4));
    end
    maxLag = opts.maxLag;
    if maxLag < 1
        error('validation:autocorr:badMaxLag', ...
              'maxLag must be ≥ 1; got %d.', maxLag);
    end

    % Sample autocorrelation per eq. (1)
    eps_centred = eps - mean(eps);
    varE        = mean(eps_centred .^ 2);          % N-divisor sample variance

    R = zeros(maxLag + 1, 1);
    R(1) = 1;  % ̂Rhat(0) ≡ 1 by construction (the convention from L11 slide 11)

    if varE > 0
        for tau = 1 : maxLag
            num   = sum( eps_centred(1 : N - tau) .* eps_centred(1 + tau : N) ) / N;
            R(tau + 1) = num / varE;
        end
    else
        % Pathological case: epsilon is exactly constant => variance 0.
        warning('validation:autocorr:zeroVariance', ...
                'Residual has zero variance - R̂(τ ≥ 1) set to 0.');
    end

    % 95% white-noise band per eq. (3)
    band = 1.96 / sqrt(N);

    % Whiteness score per eq. (4)
    inBand    = abs(R(2:end)) <= band;
    pctInBand = 100 * sum(inBand) / numel(inBand);

    % Pack output
    out.lags      = (0 : maxLag).';
    out.R         = R;
    out.band      = band;
    out.pctInBand = pctInBand;
    out.maxLag    = maxLag;
    out.N         = N;
    out.label     = opts.label;
end
