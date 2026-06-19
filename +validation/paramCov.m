function out = paramCov(A, residuals, opts)
% Inputs: A is the regression matrix, residuals are model errors, opts can include theta.
% Outputs: out stores covariance, standard errors, t-stats and conditioning.
% PARAMCOV Statistical analysis of OLS parameter estimates: covariance,

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'label'),   opts.label   = '';   end
    if ~isfield(opts, 'theta'),   opts.theta   = [];   end
    if ~isfield(opts, 'tThresh'), opts.tThresh = 2;    end

    residuals = residuals(:);
    [N, p]    = size(A);

    if numel(residuals) ~= N
        error('validation:paramCov:sizeMismatch', ...
              'A has %d rows but residuals has %d. Must match.', ...
              N, numel(residuals));
    end

    % Residual variance σ̂^2 (eq. 2)
    dof = N - p;
    if dof < 1
        warning('validation:paramCov:noDOF', ...
                'N = %d, p = %d - no residual degrees of freedom (N ≤ p).', N, p);
        sigma2 = NaN;
    else
        sigma2 = (residuals.' * residuals) / dof;
    end

    % (AᵀA)⁻¹ with pinv fallback
    AtA      = A.' * A;
    condAtA  = cond(AtA);
    usedPinv = false;
    try
        invAtA = AtA \ eye(p);
        % MATLAB will warn (but not error) on rank-deficient AtA. Force the
        if any(~isfinite(invAtA(:)))
            error('validation:paramCov:nonFiniteInverse', ...
                  'inv(AᵀA) has non-finite entries - falling back to pinv.');
        end
    catch %#ok<CTCH>
        invAtA  = pinv(AtA);
        usedPinv = true;
        warning('validation:paramCov:usedPinv', ...
                'AᵀA is rank-deficient or ill-conditioned (κ=%.2e); used pinv. SEs are lower bounds.', condAtA);
    end

    % Cov, SE, t-stat (eqs. 3-5)
    Cov = sigma2 * invAtA;
    SE  = sqrt(max(diag(Cov), 0));   % guard against tiny negatives from round-off

    if ~isempty(opts.theta)
        theta = opts.theta(:);
        if numel(theta) ~= p
            error('validation:paramCov:thetaSizeMismatch', ...
                  'opts.theta has %d elements but A has %d columns.', ...
                  numel(theta), p);
        end
        tStat   = theta ./ SE;             % may have ±Inf when SE = 0
        flagged = abs(tStat) < opts.tThresh;
    else
        theta   = [];
        tStat   = [];
        flagged = [];
    end

    % Pack output
    out.Cov          = Cov;
    out.SE           = SE;
    out.tStat        = tStat;
    out.sigma2       = sigma2;
    out.dof          = dof;
    out.flagged      = flagged;
    out.condAtA      = condAtA;
    out.usedPinv     = usedPinv;
    out.bnetVariance = diag(Cov);     % alias of SE.^2; see header note
    out.theta        = theta;
    out.label        = opts.label;
    out.N            = N;
    out.p            = p;
end
