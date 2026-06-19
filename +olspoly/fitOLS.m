function out = fitOLS(A, Y)
% FITOLS Closed-form Ordinary Least Squares estimate.
% Inputs: A is the regression matrix, Y is the measured output vector.
% Outputs: out stores theta, predictions, residuals, RMS, covariance and sizes.

    [N, p] = size(A);
    if size(Y,1) ~= N
        error('olspoly:fitOLS:sizeMismatch', ...
              'A has %d rows but Y has %d. They must match.', N, size(Y,1));
    end

    % Stable solve (QR-based under the hood).
    theta = A \ Y;

    Yhat       = A * theta;
    residuals  = Y - Yhat;
    sigma2hat  = (residuals.' * residuals) / max(1, N - p);
    RMS        = sqrt( mean(residuals.^2) );

    % Cov{θ̂} = σ̂^2 · (AᵀA)⁻¹. Only computed here; not used downstream by
    AtA = A.' * A;
    try
        CovTheta = sigma2hat * (AtA \ eye(p));
    catch %#ok<CTCH>
        warning('olspoly:fitOLS:singularAtA', ...
                'AᵀA is singular at p=%d; returning pinv-based covariance.', p);
        CovTheta = sigma2hat * pinv(AtA);
    end
    condAtA = cond(AtA);

    out.theta      = theta;
    out.Yhat       = Yhat;
    out.residuals  = residuals;
    out.RMS        = RMS;
    out.sigma2hat  = sigma2hat;
    out.CovTheta   = CovTheta;
    out.condAtA    = condAtA;
    out.p          = p;
    out.N          = N;
end
