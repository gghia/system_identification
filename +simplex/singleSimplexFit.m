function out = singleSimplexFit(X, Y, V, T, d)
% Inputs: X/Y are data, V/T define one simplex, d is the polynomial degree.
% Outputs: out stores coefficients, predictions, residuals, covariance and basis data.
% SINGLESIMPLEXFIT Part 2 OLS fit of the degree-d B-form polynomial on one

    % Input shaping
    if isempty(T), T = [1, 2, 3]; end          % default single triangle
    if size(T, 1) ~= 1
        error('simplex:singleSimplexFit:multiSimplex', ...
              ['T has %d rows; this function fits ONE simplex. ', ...
               'For Part 3 use +spline/ecolsKKT.m or ecolsNullspace.m.'], ...
              size(T, 1));
    end

    [N, n] = size(X);
    if size(Y, 1) ~= N
        error('simplex:singleSimplexFit:sizeMismatch', ...
              'X has %d rows but Y has %d. They must match.', N, size(Y, 1));
    end
    if size(V, 2) ~= n
        error('simplex:singleSimplexFit:dimMismatch', ...
              'V has %d cols but X has %d. They must match.', size(V, 2), n);
    end
    if size(V, 1) ~= n + 1
        error('simplex:singleSimplexFit:badV', ...
              'V must have n+1 = %d rows for a single n-simplex (got %d).', ...
              n + 1, size(V, 1));
    end
    if d < 1
        error('simplex:singleSimplexFit:badDegree', ...
              'd must be ≥ 1 (got %d).', d);
    end

    % Step 4-5: membership + cartesian to barycentric
    [bary, simplexIdx] = simplex.cart2bary(V, T, X);

    % Sanity warning if any data point lies outside the triangle.
    nOut = sum(isnan(simplexIdx));
    if nOut > 0
        warning('simplex:singleSimplexFit:outOfHull', ...
                ['%d / %d data point(s) lie OUTSIDE the bounding triangle ', ...
                 '(negative barycentric coordinate). They are still ', ...
                 'included in the OLS fit, but the B-coefficient bound ', ...
                 'min(ĉ) ≤ p ≤ max(ĉ) no longer applies on those samples ', ...
                 '(L11 slides 18-19).'], nOut, N);
    end

    % Step 6-7: multi-index + basis matrix
    [kappa, multinomCoef] = simplex.multiIndex(n, d);    % d̂ × (n+1), d̂ × 1
    B = simplex.bcoefBasis(bary, kappa, multinomCoef);   % N × d̂

    % quick partition-of-unity sanity check
    rowSums = sum(B, 2);
    rowOK   = ~isnan(simplexIdx);   % only in-hull rows should sum to 1
    if any(rowOK) && any(abs(rowSums(rowOK) - 1) > 1e-9)
        warning('simplex:singleSimplexFit:partitionUnity', ...
                ['Row sums of regression matrix B deviate from 1 ', ...
                 '(worst |sum-1| = %.3g). Likely cause: dropped d!/κ! ', ...
                 'in bcoefBasis (most likely a missing d!/kappa! factor).'], ...
                max(abs(rowSums(rowOK) - 1)));
    end

    dHat = size(B, 2);

    % Step 8: OLS solve
    c    = B \ Y;
    Yhat = B * c;
    residuals = Y - Yhat;
    sigma2hat = (residuals.' * residuals) / max(1, N - dHat);
    RMS  = sqrt(mean(residuals.^2));

    % Cov{ĉ} = σ̂^2·(BᵀB)⁻¹ (eq. 6).
    BtB = B.' * B;
    try
        Covc = sigma2hat * (BtB \ eye(dHat));
    catch %#ok<CTCH>
        warning('simplex:singleSimplexFit:singularBtB', ...
                'BᵀB is singular at d̂=%d; returning pinv-based Cov{ĉ}.', dHat);
        Covc = sigma2hat * pinv(BtB);
    end
    condBtB = cond(BtB);

    % Pack output
    out.c            = c;
    out.Yhat         = Yhat;
    out.residuals    = residuals;
    out.RMS          = RMS;
    out.sigma2hat    = sigma2hat;
    out.Covc         = Covc;
    out.condBtB      = condBtB;
    out.kappa        = kappa;
    out.multinomCoef = multinomCoef;
    out.B            = B;          % regression matrix
    out.d            = d;
    out.bary         = bary;
    out.simplexIdx   = simplexIdx;
    out.cBounds      = [min(c),  max(c)];
    out.yRange       = [min(Y),  max(Y)];
end
