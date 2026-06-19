%% RUN_PART2 - Part 2 driver
% Part 2 reuses the reconstructed signals saved by Part 1, and Part 3 reads the same file.

clear; clc; close all;

% Force a light (white) figure theme so saved PNGs have a white background
try figCfg = settings; figCfg.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light'; 
catch
end
%% Setup
codeRoot    = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeRoot);
figDir      = fullfile(codeRoot, 'p2_figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end
addpath(projectRoot);

fprintf('===== Part 2 driver =====\n\n');

%% POINT 1 Load reconstructed data from Part 1
% The loaded struct contains the state-reconstructed signals from Part 1.

reconFile = fullfile(codeRoot, 'recon.mat');
if ~isfile(reconFile)
    error('run_part2:noRecon', ...
          'recon.mat not found in %s. Run run_part1 first.', codeRoot);
end
R = load(reconFile);
fprintf('2.1  Loaded recon.mat: %d samples, Ĉ_α_up_final = %.5f.\n', ...
        numel(R.Cm), R.Cfinal);

X = [R.alpha_true, R.beta_m];   % Regressor input without V.
Y = R.Cm;
N = numel(Y);

Xid  = X(R.idMask, :);  Yid  = Y(R.idMask);
Xval = X(R.valMask,:);  Yval = Y(R.valMask);

%% POINT 2 B-form polynomial formulation
% Multi-index k = (k_0, k_1, k_2) with |k| = d.

fprintf('2.2  B-form polynomial: see +simplex/multiIndex.m and +simplex/bcoefBasis.m headers.\n\n');

%% POINT 3 Triangulation, single bounding triangle
% Pick three vertices that form a triangle enclosing the data cloud with a small margin.

fprintf('2.3  Building bounding triangle around (α_true, β_m) data...\n');

% TIGHT bounding 2-simplex: right triangle (right angle at lower-left apex v0,
% one cathetus along +alpha to v1, one along +beta to v2) with legs sized to
% the DATA, not to a fixed multiple of the bbox. Anchor v0 a 1% pad below-left
% of the data min corner; find the TANGENT leg scale k_tan (smallest such
% triangle containing all data: max over data of [(a-x0)/ax+(b-y0)/ay]=1);
% then step back by a safety factor sf so every point is strictly interior.
% The old 'bounding' was much larger, it made legs 2.4x the data extent -> ~30% area coverage,
% tiny barycentric box -> near-collinear B-form columns -> kappa(B'B)~4e9,
% coefs ~[-74,153]. A data-tight triangle (~50% coverage) cuts kappa ~25x and
% the coef range ~6.5x at IDENTICAL fit (degree-d B-form spans the same
% polynomial space regardless of triangle size).
panch = 0.01;                                    % padding 1% of data extent
sf    = 1.08;                                    % safety margin
xmn = min(X(:,1)); xmx = max(X(:,1)); dxX = xmx - xmn;
ymn = min(X(:,2)); ymx = max(X(:,2)); dyX = ymx - ymn;
x0 = xmn - panch*dxX;   y0 = ymn - panch*dyX;    % lower-left right-angle apex
axx = xmx - x0;         ayy = ymx - y0;
ktan = max( (X(:,1)-x0)/axx + (X(:,2)-y0)/ayy ); % tangent leg scale
kk   = sf * ktan;
V_p2 = [ x0,          y0;                         % v0  (right-angle apex)
         x0 + kk*axx, y0;                         % v1  (alpha leg)
         x0,          y0 + kk*ayy ];              % v2  (beta leg)
T_p2 = [1 2 3];
fprintf('    Triangle vertices:\n');
disp(V_p2);

% Sanity check: convert all data to barycentric coordinates and verify that they stay nonnegative.
bary_all = simplex.cart2bary(V_p2, T_p2, X);
nBad = sum(any(bary_all < -1e-10, 2));
if nBad > 0
    warning('run_part2:outOfTri', ...
            '%d / %d points have negative barycentric coords - enlarge margin.', ...
            nBad, N);
else
    fprintf('    All %d points lie inside the bounding triangle.\n', N);
end

% Visualise the triangle together with the data cloud and the chosen vertices.
figure('Name','P2.2 triangulation','NumberTitle','off');
plot(X(:,1), X(:,2), '.', 'Color', [.6 .6 .6], 'MarkerSize', 3); hold on;
% Close the triangle by repeating the first vertex.
patchX = V_p2([1 2 3 1], 1);
patchY = V_p2([1 2 3 1], 2);
plot(patchX, patchY, '-r', 'LineWidth', 1.5);
plot(V_p2(:,1), V_p2(:,2), 'or', 'MarkerFaceColor','r');
for j = 1:3
    text(V_p2(j,1), V_p2(j,2), sprintf('  v_%d', j-1), 'Color','r');
end
grid on; axis equal;
xlabel('\alpha_{true}'); ylabel('\beta_m');
title('Data-tight 2-simplex around the SysID scatter');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p22_triangle.png'), 'Resolution', 200);

%% POINT 4 OLS estimator + degree sweep
% For each d in {2,...,30}, build B and solve the coefficient vector c.

fprintf('\n2.4  Single-simplex B-form OLS sweep d = 2..30...\n');

degSweep_p2 = 2:30;
RMS_id_p2   = zeros(size(degSweep_p2));
RMS_val_p2  = zeros(size(degSweep_p2));
condBtB_p2  = zeros(size(degSweep_p2));
simpOuts    = cell(size(degSweep_p2));

for k = 1:numel(degSweep_p2)
    d_k = degSweep_p2(k);
    fit = simplex.singleSimplexFit(Xid, Yid, V_p2, T_p2, d_k);

    % Build the validation design matrix to evaluate the validation RMS.
    bary_val = simplex.cart2bary(V_p2, T_p2, Xval);
    Bv       = simplex.bcoefBasis(bary_val, fit.kappa, fit.multinomCoef);
    Yhat_val = Bv * fit.c;

    RMS_id_p2(k)  = fit.RMS;
    RMS_val_p2(k) = sqrt( mean( (Yval - Yhat_val).^2 ) );
    condBtB_p2(k) = fit.condBtB;
    simpOuts{k}   = fit;
end

% Model selection: lowest in-domain validation RMS over the inspected sweep.
condBound2  = 1/eps;
[~, kStar2] = min(RMS_val_p2);
dStar2      = degSweep_p2(kStar2);
fprintf('    Chosen d* = %d  (RMS_val = %.4g, kappa(B''B) = %.2e; selected by in-domain validation).\n', ...
        dStar2, RMS_val_p2(kStar2), condBtB_p2(kStar2));
condCut = find(condBtB_p2 >= condBound2, 1, 'first');
if ~isempty(condCut)
    fprintf('    kappa(B''B) first exceeds 1/eps at d=%d.\n', degSweep_p2(condCut));
end

% Plot the identification and validation RMS curves versus d.
figure('Name','P2.3 single-simplex sweep','NumberTitle','off');
subplot(2,1,1);
semilogy(degSweep_p2, RMS_id_p2, '-o', degSweep_p2, RMS_val_p2, '-s'); grid on;
xlabel('degree d'); ylabel('RMS');
legend('identification','validation','Location','best');
title('Single-simplex B-form fit - RMS vs. d');
hStarRef = xline(dStar2, '--k');
hStarRef.Annotation.LegendInformation.IconDisplayStyle = 'off';
subplot(2,1,2);
semilogy(degSweep_p2, condBtB_p2, '-o'); hold on;
hCondRef = yline(condBound2, '--k', '1/\epsilon', 'LabelHorizontalAlignment','left');
hCondRef.Annotation.LegendInformation.IconDisplayStyle = 'off';
grid on;
xlabel('degree d'); ylabel('\kappa(B^TB)');
title('Cond(BᵀB) - high-degree single simplex becomes ill-conditioned');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p23_sweep.png'), 'Resolution', 200);

bestFit2 = simpOuts{kStar2};

%% POINT 5 Performance + statistical validation
% Reusing the Part 1 diagnostics, then add the B-coefficient bound check and the special-validation summary.

fprintf('\n2.5  Validating d* = %d ...\n', dStar2);

resOut2 = validation.residuals(Yid, bestFit2.Yhat, Xid);
acOut2  = validation.autocorr(resOut2.eps);
% Pass theta=c so paramCov can compute t-statistics and significance flags.
pcOut2  = validation.paramCov(bestFit2.B, resOut2.eps, struct('theta', bestFit2.c));

fprintf('    Residual RMS:     %.4g\n', resOut2.RMS);
fprintf('    Residual mean:    %.3g\n', resOut2.mean);
fprintf('    Autocorr in band: %.1f%%\n', acOut2.pctInBand);

% Check that min(c) <= p(x) <= max(c) inside the simplex.
yRange = [min(Yid), max(Yid)];
cRange = [min(bestFit2.c), max(bestFit2.c)];
fprintf('    y range:     [%.3g, %.3g]\n', yRange(1), yRange(2));
fprintf('    ĉ range:    [%.3g, %.3g]\n', cRange(1), cRange(2));
if cRange(1) < yRange(1) - 0.5*range(yRange) || cRange(2) > yRange(2) + 0.5*range(yRange)
    fprintf('    WARN: B-coefs exceed data range by >50%% - possible over-fit.\n');
else
    fprintf('    OK: ĉ within reasonable distance of data range.\n');
end

% Plot the B-net by colouring each spatial anchor with its B-coefficient value.
anchors_bary = bestFit2.kappa ./ dStar2;
anchors_cart = bsplinen_bary2cart(V_p2, anchors_bary);
figure('Name','P2.4 B-net','NumberTitle','off');
plot(patchX, patchY, '-k'); hold on; grid on; axis equal;
scatter(anchors_cart(:,1), anchors_cart(:,2), 80, bestFit2.c, 'filled');
colorbar; colormap(parula);
xlabel('\alpha_{true}'); ylabel('\beta_m');
title(sprintf('B-net for d*=%d - colour = ĉ_k', dStar2));
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p24_bnet.png'), 'Resolution', 200);

figure('Name','P2.4 B-coef variance','NumberTitle','off');
plot(patchX, patchY, '-k'); hold on; grid on; axis equal;
scatter(anchors_cart(:,1), anchors_cart(:,2), 80, diag(bestFit2.Covc), 'filled');
colorbar;
xlabel('\alpha_{true}'); ylabel('\beta_m');
title('B-coefficient variance at spatial anchors k/d');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p24_var.png'), 'Resolution', 200);

%% POINT 6 - Special-validation RMS

fprintf('\n2.6  Special-validation RMS...\n');

S = data.loadSpecial(projectRoot);

% Given Xq, transform to barycentric coordinates on this triangle and evaluate the B-form.
predFun_p2 = @(Xq) predictSimplex(Xq, V_p2, T_p2, bestFit2);
% Note: hullData must be the identification slice Xid, not the full X.
specOut_p2 = validation.specialRMS(predFun_p2, S.alpha_val, S.beta_val, ...
                                    S.Cm_val, Xid);
fprintf('    %s\n', specOut_p2.summary);

%% Save outputs

p2_simplex.dStar       = dStar2;
p2_simplex.c           = bestFit2.c;
p2_simplex.kappa       = bestFit2.kappa;
p2_simplex.V           = V_p2;
p2_simplex.T           = T_p2;
p2_simplex.RMS_id      = RMS_id_p2;
p2_simplex.RMS_val     = RMS_val_p2;
p2_simplex.condBtB     = condBtB_p2;
p2_simplex.specOut     = specOut_p2;
p2_simplex.resOut      = resOut2;
p2_simplex.acOut       = acOut2;
p2_simplex.pcOut       = pcOut2;
save(fullfile(codeRoot, 'p2_simplex.mat'), '-struct', 'p2_simplex');
fprintf('\nSaved p2_simplex.mat (used by run_part4).\n');

fprintf('\n===== Part 2 complete =====\n');

%% Local predictor helper
function Yq = predictSimplex(Xq, V, T, fit)
% Inputs: Xq are query points, V/T define the simplex, fit stores c and basis data.
% Outputs: Yq is the predicted Cm at the query points.
% Evaluate the single-simplex B-form fit at the requested query points.
    bary = simplex.cart2bary(V, T, Xq);
    Bv   = simplex.bcoefBasis(bary, fit.kappa, fit.multinomCoef);
    Yq   = Bv * fit.c;
% Points outside the triangle have at least one negative barycentric coordinate.
end
