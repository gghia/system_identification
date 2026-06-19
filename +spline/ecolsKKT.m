function out = ecolsKKT(Bglobal, H, Y)
% Inputs: Bglobal is the regression matrix, H*c=0 are constraints, Y is data.
% Outputs: out stores constrained coefficients, residuals, covariance and diagnostics.
% ECOLSKKT Equality-Constrained OLS via the Lagrangian / KKT system.

    [N, p] = size(Bglobal);
    if numel(Y) ~= N
        error('spline:ecolsKKT:sizeMismatch', ...
              'Bglobal has %d rows; Y has %d. They must match.', N, numel(Y));
    end

    % Handle the "no constraints" degenerate case
    if isempty(H) || size(H, 1) == 0
        % Plain OLS via backslash. Cov, sigma2, RMS are still reported.
        c       = Bglobal \ Y;
        lambda  = zeros(0, 1);
        Yhat    = Bglobal * c;
        resid   = Y - Yhat;
        RMS     = sqrt(mean(resid.^2));
        sig2    = (resid.' * resid) / max(1, N - p);
        AtA     = Bglobal.' * Bglobal;
        try
            Covc = sig2 * (AtA \ speye(p));
        catch %#ok<CTCH>
            Covc = sig2 * pinv(full(AtA));
        end
        out.c              = c;
        out.lambda         = lambda;
        out.Yhat           = Yhat;
        out.residuals      = resid;
        out.RMS            = RMS;
        out.sigma2hat      = sig2;
        out.Covc           = full(Covc);
        out.cond           = condest(AtA);
        out.constraintViol = 0;
        out.usedPinv       = false;
        return;
    end

    m = size(H, 1);
    if size(H, 2) ~= p
        error('spline:ecolsKKT:sizeMismatch', ...
              'H has %d cols; Bglobal has %d cols.', size(H,2), p);
    end

    % Assemble normal-equation pieces
    BtB = Bglobal.' * Bglobal;          % p x p,  sparse, symmetric PSD
    BtY = Bglobal.' * Y;                % p x 1

    % KKT matrix (eq. 3) and rhs.
    K   = [ BtB,            H.';
            H,              sparse(m, m) ];
    rhs = [ BtY;
            zeros(m, 1) ];

    % Solve
    usedPinv = false;
    lastwarn('');
    sol = K \ rhs;
    [warnMsg, ~] = lastwarn;
    if ~isempty(warnMsg) || any(~isfinite(sol))
        Kfull    = full(K);
        Kinv     = pinv(Kfull);
        sol      = Kinv * rhs;
        usedPinv = true;
    end

    c      = sol(1:p);
    lambda = sol(p + 1 : p + m);

    % Extract C_1 for Cov(chat) (eq. 4-5)
    if usedPinv
        C1 = Kinv(1:p, 1:p);
    else
        % K X = [I; 0] to get the first p columns of K^-1 in one go.
        E1 = [speye(p); sparse(m, p)];
        try
            X  = K \ E1;
            C1 = full(X(1:p, :));
        catch %#ok<CTCH>
            Kinv = pinv(full(K));
            C1   = Kinv(1:p, 1:p);
            usedPinv = true;
        end
    end

    % Residuals, sigma^2, Covariance
    Yhat  = Bglobal * c;
    resid = Y - Yhat;
    RMS   = sqrt(mean(resid.^2));

    % Effective degrees of freedom under constraints: N_free = p - rank(H).
    rH    = rank(full(H));
    Nfree = p - rH;
    sig2  = (resid.' * resid) / max(1, N - Nfree);

    Covc            = sig2 * C1;
    constraintViol  = norm(H * c);

    % cond(K) is finite if K is non-singular, Inf otherwise.
    out.cond        = condest(K);

    out.c              = c;
    out.lambda         = lambda;
    out.Yhat           = Yhat;
    out.residuals      = resid;
    out.RMS            = RMS;
    out.sigma2hat      = sig2;
    out.Covc           = Covc;
    out.constraintViol = constraintViol;
    out.usedPinv       = usedPinv;
end
