 function out = ecolsNullspace(Bglobal, H, Y)
% Inputs: Bglobal is the regression matrix, H*c=0 are constraints, Y is data.
% Outputs: out stores the null-space fit, Gamma, residuals and diagnostics.
% ECOLSNULLSPACE Equality-Constrained OLS via null-space reparameterisation.

    [N, p] = size(Bglobal);
    if numel(Y) ~= N
        error('spline:ecolsNullspace:sizeMismatch', ...
              'Bglobal has %d rows; Y has %d.', N, numel(Y));
    end

    % Build null(H)
    if isempty(H) || size(H, 1) == 0
        % If H has no rows, there are no equations H*c = 0 to satisfy.
        % That means every coefficient vector c is allowed. With
        % Gamma = eye(p), the reduced variable is just c itself, so this
        % becomes the usual least-squares fit on Bglobal.
        Gamma = eye(p);
    else
        if size(H, 2) ~= p
            error('spline:ecolsNullspace:sizeMismatch', ...
                  'H has %d cols; Bglobal has %d cols.', size(H,2), p);
        end
        % `null(H)` returns an ORTHONORMAL basis via SVD.
        Gamma = null(full(H));
    end

    Nfree = size(Gamma, 2);

    % Solve the reduced OLS
    BgGamma = Bglobal * Gamma;          % N x N_free   (sparse * dense = dense
                                        % unless N_free is tiny)
    cTilde  = BgGamma \ Y;              % eq. (5)

    % Recover global c from the reduced coefficients.
    c = Gamma * cTilde;

    % Residuals + statistics
    Yhat  = Bglobal * c;
    resid = Y - Yhat;
    RMS   = sqrt(mean(resid.^2));
    sig2  = (resid.' * resid) / max(1, N - Nfree);

    % Covariance of cTilde: sig2 * inv(BgGamma' * BgGamma).
    GtG = BgGamma.' * BgGamma;
    try
        invGtG = GtG \ eye(Nfree);
    catch %#ok<CTCH>
        warning('spline:ecolsNullspace:singularReduced', ...
                'B · Γ is rank-deficient. Falling back to pinv for covariance.');
        invGtG = pinv(full(GtG));
    end
    CovcTilde = sig2 * invGtG;
    Covc      = Gamma * CovcTilde * Gamma.';

    out.c              = c;
    out.cTilde         = cTilde;
    out.Gamma          = Gamma;
    out.Nfree          = Nfree;
    out.Yhat           = Yhat;
    out.residuals      = resid;
    out.RMS            = RMS;
    out.sigma2hat      = sig2;
    out.Covc           = Covc;
    out.condBgGamma    = cond(BgGamma);
    out.constraintViol = norm(full(H) * c);  % full() because H may be sparse
end
