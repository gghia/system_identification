%% RUN_PART3 Part 3 driver
% Multivariate simplex B-spline fit (the heart of the assignment).

clear; clc; close all;

% Force a light (white) figure theme so saved PNGs have a white background.

try
    figCfg = settings; figCfg.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light'; 
catch
end
%% Setup
codeRoot    = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeRoot);
figDir      = fullfile(codeRoot, 'p3_figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end
addpath(projectRoot);   % Add the shared barycentric conversion helpers to the path.

fprintf('===== Part 3 driver =====\n\n');

%% POINT 1 Load data + block diagram
% Reproduce this in the report as a boxes-and-arrows figure, with the two-stage structure kept explicit.

reconFile = fullfile(codeRoot, 'recon.mat');
if ~isfile(reconFile)
    error('run_part3:noRecon', 'recon.mat not found. Run run_part1 first.');
end
R = load(reconFile);
fprintf('3.1  Loaded recon.mat: %d samples.\n', numel(R.Cm));

X = [R.alpha_true, R.beta_m];
Y = R.Cm;

Xid  = X(R.idMask,  :);  Yid  = Y(R.idMask);
Xval = X(R.valMask, :);  Yval = Y(R.valMask);

% Print a text version of the block diagram for the report narrative.
fprintf('\n    Block diagram (sketch in the report):\n');
fprintf('      raw measurements → IEKF → reconstructed (α_true, β, V, Cm)\n');
fprintf('                                  ↓\n');
fprintf('              triangulation (V, T) → cart→bary (tsearchn)\n');
fprintf('                                  ↓\n');
fprintf('              buildRegression B    + smoothnessMatrix H(r)\n');
fprintf('                                  ↓\n');
fprintf('              EC-OLS: KKT or null-space → ĉ\n');
fprintf('                                  ↓\n');
fprintf('              residual / param-cov / B-coef-bound validation\n\n');

%% POINT 2 Triangulation strategy (part of P3.2)
% Delaunay on a regular vertex grid over the (α, β) box.

fprintf('3.2  Building triangulation (Delaunay on a 4×4 vertex grid)...\n');

[V_p3, T_p3] = simplex.triangulate(X, 'delaunay', struct('nx', 4, 'ny', 4, 'margin', 0.02));
T_count = size(T_p3, 1);
fprintf('    %d vertices, %d simplices.\n', size(V_p3, 1), T_count);

% Visualise the triangulation together with the data cloud.
figure('Name','P3.2 triangulation','NumberTitle','off');
triplot(T_p3, V_p3(:,1), V_p3(:,2), 'k'); hold on;
plot(X(:,1), X(:,2), '.', 'Color',[.6 .6 .6], 'MarkerSize', 2);
grid on; axis equal;
xlabel('\alpha_{true}'); ylabel('\beta_m');
title(sprintf('Triangulation: T = %d simplices', T_count));
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p32_triangulation.png'), 'Resolution', 200);

%% POINT 3 Smoothness matrix
% Build H for the chosen continuity order r.

fprintf('\n3.3  Building smoothness matrix H for d=4, r=0 (initial pass)...\n');

d_init = 4;
r_init = 0;

[H_init, info_init] = spline.smoothnessMatrix(V_p3, T_p3, d_init, r_init);
fprintf('    H: %d rows × %d cols, rank %d. Continuity rows per edge: %d.\n', ...
        size(H_init,1), size(H_init,2), rank(full(H_init)), info_init.rowsPerEdge);

%% POINT 4 EC-OLS estimators KKT + null-space
% Both estimators are presented in the report.

fprintf('\n3.4  Building regression matrix B and fitting EC-OLS (KKT + nullspace)...\n');

[Bid, info_B] = spline.buildRegression(Xid, V_p3, T_p3, d_init);
fprintf('    B: %d × %d, in-hull samples: %d / %d.\n', ...
        size(Bid,1), size(Bid,2), sum(info_B.inHullMask), numel(Yid));
Yid_in = Yid(info_B.inHullMask);

% Compute both estimators and cross-check them against each other.
outKKT  = spline.ecolsKKT(Bid, H_init, Yid_in);
outNULL = spline.ecolsNullspace(Bid, H_init, Yid_in);

xcheck = norm(outKKT.c - outNULL.c) / max(norm(outNULL.c), 1e-30);
fprintf('    KKT vs null-space relative cross-check: %.3e (target < 1e-6).\n', xcheck);
if xcheck > 1e-6
    warning('run_part3:ecolsMismatch', ...
            'KKT and null-space disagree (%.3e). Likely H is rank-deficient and the KKT pinv fallback is biased - null-space ĉ is more reliable here.', xcheck);
end

fprintf('    Identification RMS - KKT: %.4g, null-space: %.4g.\n', outKKT.RMS, outNULL.RMS);
fprintf('    Free degrees of freedom (null(H)): %d / %d.\n', outNULL.Nfree, size(Bid,2));

%% POINT 5 Sweep (d, r, T)
% d \isin {2,3,4,5}; r \isin {0,1} (require r < d); T \isin {2, 8, 18, 32}.

fprintf('\n3.5  Sweep (d, r, T)...\n');

degs = [2 3 4 5];           % d
rs   = [0 1];               % r  (r < d required)
Tgrids = [2, 8, 18, 32];    % approximate target T (via nx×ny grids)

% Each sweep row stores [d, r, T_actual, RMS_id, RMS_val, ...].
sweepTable = [];

predFuns = cell(0);     % keep prediction functions for 3.6 / 3.7
sweepFits = cell(0);
sweepLabels = cell(0);

for tg = Tgrids
    % Build a grid that yields ~tg simplices via Delaunay
    nside = max(2, round(1 + sqrt(tg/2)));
    if tg == 2
        % Special case: use the rectangle split into two simplices when tg = 2.
        margin = 0.02;
        xL = min(X(:,1)) - margin * range(X(:,1));
        xH = max(X(:,1)) + margin * range(X(:,1));
        yL = min(X(:,2)) - margin * range(X(:,2));
        yH = max(X(:,2)) + margin * range(X(:,2));
        V_t = [xL yL; xH yL; xH yH; xL yH];
        T_t = [1 2 3; 1 3 4];
    else
        [V_t, T_t] = simplex.triangulate(X, 'delaunay', ...
                      struct('nx', nside, 'ny', nside, 'margin', 0.02));
    end
    Tact = size(T_t, 1);

    [B_id_t, info_id_t] = spline.buildRegression(Xid, V_t, T_t, max(degs));
    % Rebuild the matrix below for each d; this pass only records info_id_t.inHullMask.

    for d_k = degs
        for r_k = rs
            if r_k >= d_k, continue; end

            [B_id_k, info_id_k] = spline.buildRegression(Xid, V_t, T_t, d_k);
            Y_id_in = Yid(info_id_k.inHullMask);

            % Build the smoothness matrix.
            [H_k, ~] = spline.smoothnessMatrix(V_t, T_t, d_k, r_k);

            % Use the null-space estimator by default because it is numerically more stable.
            tStart = tic;
            try
                ec = spline.ecolsNullspace(B_id_k, H_k, Y_id_in);
            catch ME
                fprintf('      [skip d=%d, r=%d, T=%d] %s\n', d_k, r_k, Tact, ME.message);
                continue;
            end
            fitTime = toc(tStart);

            % Compute the validation RMS for this configuration.
            [B_val_k, info_val_k] = spline.buildRegression(Xval, V_t, T_t, d_k);
            Y_val_in = Yval(info_val_k.inHullMask);
            RMS_val_k = sqrt( mean( (Y_val_in - B_val_k * ec.c).^2 ) );

            sweepTable(end+1, :) = [d_k, r_k, Tact, ec.RMS, RMS_val_k, ...
                                     ec.condBgGamma, ec.Nfree, fitTime];
            sweepLabels{end+1} = sprintf('d=%d r=%d T=%d', d_k, r_k, Tact);
            sweepFits{end+1} = struct('c', ec.c, 'V', V_t, 'T', T_t, ...
                                      'd', d_k, 'r', r_k, ...
                                      'Nfree', ec.Nfree, 'Tact', Tact);
        end
    end
end

fprintf('    Sweep done: %d configurations.\n', size(sweepTable,1));

% Print the headline table.
fprintf('\n    %-6s %-3s %-4s %-12s %-12s %-12s %-6s %-8s\n', ...
        'd','r','T','RMS_id','RMS_val','cond(BΓ)','Nfree','time[s]');
for i = 1:size(sweepTable, 1)
    row = sweepTable(i, :);
    fprintf('    %-6d %-3d %-4d %-12.4g %-12.4g %-12.3g %-6d %-8.3f\n', ...
            row(1), row(2), row(3), row(4), row(5), row(6), row(7), row(8));
end

% Select the minimum validation RMS subject to numerical reliability.
condBound = 1e10;
stableMask = sweepTable(:, 6) < condBound;
if any(stableMask)
    stableIdx = find(stableMask);
    [~, iLocal] = min(sweepTable(stableIdx, 5));
    iBest = stableIdx(iLocal);
    fprintf('\n    Stable configs (cond < %.0e): %d / %d.\n', ...
            condBound, sum(stableMask), size(sweepTable, 1));
else
    % I pick the least ill-conditioned configuration if no stable candidate exists.
    [~, iBest] = min(sweepTable(:, 6));
    fprintf('\n    WARN: no stable config; falling back to least ill-conditioned.\n');
end
fprintf('    Best STABLE validation RMS: %s with RMS_val = %.4g (cond = %.2e).\n', ...
        sweepLabels{iBest}, sweepTable(iBest, 5), sweepTable(iBest, 6));
% Also report the raw validation-RMS argmin for contrast, without the stability filter.
[~, iRaw] = min(sweepTable(:, 5));
fprintf('    (Argmin RMS_val WITHOUT stability filter: %s, RMS_val = %.4g, cond = %.2e - UNRELIABLE)\n', ...
        sweepLabels{iRaw}, sweepTable(iRaw, 5), sweepTable(iRaw, 6));
bestFit3 = sweepFits{iBest};

% Cross-check KKT and null-space at the chosen well-conditioned configuration.
[B_best_xc, info_best_xc] = spline.buildRegression(Xid, bestFit3.V, bestFit3.T, bestFit3.d);
[H_best_xc, ~]            = spline.smoothnessMatrix(bestFit3.V, bestFit3.T, bestFit3.d, bestFit3.r);
Y_best_xc    = Yid(info_best_xc.inHullMask);
outKKT_best  = spline.ecolsKKT(B_best_xc, H_best_xc, Y_best_xc);
outNULL_best = spline.ecolsNullspace(B_best_xc, H_best_xc, Y_best_xc);
xcheckBest   = norm(outKKT_best.c - outNULL_best.c) / max(norm(outNULL_best.c), 1e-30);
rankH_best   = rank(full(H_best_xc));
fprintf(['    Chosen-config KKT vs null-space cross-check: %.3e (target < 1e-6)\n' ...
         '      B: %d×%d, H: %d×%d (rank %d), Nfree = %d, ‖Hĉ‖ = %.2e.\n'], ...
        xcheckBest, size(B_best_xc,1), size(B_best_xc,2), size(H_best_xc,1), ...
        size(H_best_xc,2), rankH_best, bestFit3.Nfree, norm(H_best_xc*outNULL_best.c));

% Plot RMS_val versus d for each (r, T) pair.
figure('Name','P3.5 sweep','NumberTitle','off');
hold on; grid on;
uniqRT = unique(sweepTable(:, [2 3]), 'rows');
for k = 1:size(uniqRT, 1)
    msk = sweepTable(:,2) == uniqRT(k,1) & sweepTable(:,3) == uniqRT(k,2);
    plot(sweepTable(msk,1), sweepTable(msk,5), '-o', ...
         'DisplayName', sprintf('r=%d, T=%d', uniqRT(k,1), uniqRT(k,2)));
end
set(gca, 'YScale', 'log');
xlabel('degree d'); ylabel('validation RMS');
legend('Location','bestoutside');
title('Spline sweep - RMS_{val} vs. (d, r, T)');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p35_sweep.png'), 'Resolution', 200);

%% POINT 6 Validation of chosen model

fprintf('\n3.6  Validating chosen spline (%s)...\n', sweepLabels{iBest});

% Compute predictions over the identification set and form residuals.
[Bid_best, infoBest] = spline.buildRegression(Xid, bestFit3.V, bestFit3.T, bestFit3.d);
Yid_best = Yid(infoBest.inHullMask);
Yhat_id  = Bid_best * bestFit3.c;
resOut3  = validation.residuals(Yid_best, Yhat_id, Xid(infoBest.inHullMask,:));
acOut3   = validation.autocorr(resOut3.eps);

fprintf('    Residual RMS:      %.4g\n', resOut3.RMS);
fprintf('    Residual mean:     %.3g  (should be ~0)\n', resOut3.mean);
fprintf('    Autocorr in band:  %.1f%%  (should be ≥ 95)\n', acOut3.pctInBand);

% Check the B-coefficient bound on each simplex.
fprintf('    B-coef range:  [%.3g, %.3g];  data range: [%.3g, %.3g].\n', ...
        min(bestFit3.c), max(bestFit3.c), min(Yid_best), max(Yid_best));

% Plot the residual diagnostics figure.
figure('Name','P3.6 spline residuals','NumberTitle','off');
subplot(2,2,1); plot(resOut3.eps); grid on; title('residual time series');
subplot(2,2,2); histogram(resOut3.eps, resOut3.histEdges); grid on; title('residual histogram');
subplot(2,2,3);
Xin = Xid(infoBest.inHullMask, :);
scatter3(Xin(:,1), Xin(:,2), resOut3.eps, 4, resOut3.eps, 'filled'); grid on; view(2);
colorbar; title('residual surface over (α, β)');
xlabel('\alpha_{true}'); ylabel('\beta_m');
subplot(2,2,4); stem(acOut3.lags, acOut3.R, 'filled'); hold on; grid on;
yline(acOut3.band,'--r'); yline(-acOut3.band,'--r');
title(sprintf('autocorr (%.0f%% in band)', acOut3.pctInBand));
sgtitle(sprintf('Spline %s - validation', sweepLabels{iBest}));
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p36_diag.png'), 'Resolution', 200);

%% POINT 7 Special-validation RMS
% Evaluate the special-validation dataset and explicitly note that the special domain lies outside the training hull.

fprintf('\n3.7  Special-validation RMS (full 100-point breakdown)...\n');

S      = data.loadSpecial(projectRoot);
Xspec  = [S.alpha_val(:), S.beta_val(:)];
Cmspec = S.Cm_val(:);

% Keep the natural B-form values at all 100 points; NaN only marks out-of-triangulation samples.
Yspec = spline.evaluate(Xspec, bestFit3.V, bestFit3.T, bestFit3.d, ...
                        bestFit3.c, struct('outOfHull','nan'));

% Split by the TRAINING-DATA convex hull (Xid), NOT the triangulation hull.
Kdata      = convhull(Xid(:,1), Xid(:,2));
inHullSpec = inpolygon(Xspec(:,1), Xspec(:,2), Xid(Kdata,1), Xid(Kdata,2));
nInTri     = sum(~isnan(Yspec));
resS       = Cmspec - Yspec;

specOut_p3         = struct();
specOut_p3.RMS_in  = sqrt(mean(resS(inHullSpec).^2));   % domain of validity (n=73)
specOut_p3.RMS_out = sqrt(mean(resS(~inHullSpec).^2));  % data-free extrapolation (n=27)
specOut_p3.RMS_all = sqrt(mean(resS.^2));               % literal P3.7 number (n=100)
specOut_p3.nIn     = sum(inHullSpec);
specOut_p3.nOut    = sum(~inHullSpec);
specOut_p3.nInTri  = nInTri;
specOut_p3.summary = sprintf(['Special grid: %d/%d inside triangulation; ' ...
    'in-hull RMS=%.4g (n=%d), out-of-hull RMS=%.4g (n=%d), FULL=%.4g (n=100).'], ...
    nInTri, numel(Cmspec), specOut_p3.RMS_in, specOut_p3.nIn, ...
    specOut_p3.RMS_out, specOut_p3.nOut, specOut_p3.RMS_all);
fprintf('    %s\n', specOut_p3.summary);
fprintf(['    [in-hull = domain of validity (best of all models); out-of-hull =\n' ...
         '     data-free corner-simplex extrapolation; full = literal P3.7 number.]\n']);

%% Save outputs

p3_spline.bestFit    = bestFit3;
p3_spline.sweepTable = sweepTable;
p3_spline.sweepLabels = sweepLabels;
p3_spline.iBest      = iBest;
p3_spline.specOut    = specOut_p3;
p3_spline.resOut     = resOut3;
p3_spline.acOut      = acOut3;
% Store the KKT versus null-space cross-check statistics.
p3_spline.kktNullCrossCheck       = xcheck;        % demo config (d=4,r=0,T=18, rank-deficient)
p3_spline.kktNullCrossCheckChosen = xcheckBest;    % chosen (5,0,8) config - the meaningful one
p3_spline.crossCheckChosen = struct('xcheck', xcheckBest, ...
    'Bsize', size(B_best_xc), 'Hsize', size(H_best_xc), 'rankH', rankH_best, ...
    'Nfree', bestFit3.Nfree, 'normHc', norm(H_best_xc*outNULL_best.c));
save(fullfile(codeRoot, 'p3_spline.mat'), '-struct', 'p3_spline');
fprintf('\nSaved p3_spline.mat (used by run_part4).\n');

fprintf('\n===== Part 3 complete =====\n');

% Use the training hull to enforce that the spline is only valid inside the data hull.
function Y = maskedEval(Xq, V, T, d, c, hullX, hullY)
% Inputs: Xq are query points, V/T/d/c define the spline, hullX/hullY define valid support.
% Outputs: Y is the spline prediction, with out-of-hull points set to NaN.
    Y = spline.evaluate(Xq, V, T, d, c, struct('outOfHull','nan'));
    outMask = ~inpolygon(Xq(:,1), Xq(:,2), hullX, hullY);
    Y(outMask) = NaN;
end
