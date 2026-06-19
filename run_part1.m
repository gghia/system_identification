%% RUN_PART1 Part 1 driver
% recon.mat = struct with the reconstructed (alpha_true, β_m, V_m, C_m)

clear; clc; close all;

% Force a light (white) figure theme so saved PNGs have a white background
% (R2025b follows the dark desktop theme by default). Session-scoped — reverts
% when MATLAB exits; the try/catch keeps it harmless on older releases.
try, figCfg = settings; figCfg.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light'; catch, end

%% Setup
codeRoot    = fileparts(mfilename('fullpath'));               % .../code
projectRoot = fileparts(codeRoot);                            % .../assignment1...
figDir      = fullfile(codeRoot, 'p1_figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end
addpath(projectRoot);

fprintf('===== Part 1 driver =====\n\n');
fprintf('Project root: %s\n', projectRoot);
fprintf('Figure dir:   %s\n\n', figDir);

%% POINT 1 Data sanity + glitch detection
% Load data with +data/loadF16.

fprintf('1.1  Loading data, detecting glitch, splitting train/val...\n');

D = data.loadF16(projectRoot);
fprintf('    Loaded %d samples at dt = %g s.\n', D.N, D.dt);

% Plot raw Cm (glitch should be visually obvious)
figure('Name','P1.1 raw Cm time series','NumberTitle','off');
plot(D.Cm, '.'); grid on;
xlabel('sample index k'); ylabel('C_m raw');
title('Raw measured C_m - note the glitch band');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p11_raw_Cm.png'), 'Resolution', 200);

% Flag the glitch by hand-verified sample intervals
[keepMask, glitch] = data.detectGlitch(D.Cm, struct('intervals','hand','policy','drop'));
fprintf('    %s\n', glitch.summary);

% Overlay the flagged bursts on the raw Cm plot
figure('Name','P1.1 glitch detection','NumberTitle','off');
plot(D.Cm, '.', 'Color',[.5 .5 .5]); hold on; grid on;
plot(find(~keepMask), D.Cm(~keepMask), 'r.', 'MarkerSize', 6);
yline(glitch.medianValue, '--', 'baseline median(C_m)');
xlabel('sample index k'); ylabel('C_m');
title(sprintf('Glitch detection - %d hand-verified bursts dropped', ...
              size(glitch.intervals,1)));
legend('keep','flagged (dropped)','Location','best');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p11_glitch.png'), 'Resolution', 200);


% 3-D scatter: (alpha_m, beta_m, Cm) with the glitched samples in red
figure('Name','P1.1 3D scatter, glitch highlighted','NumberTitle','off');
plot3(D.alpha_m(keepMask),  D.beta_m(keepMask),  D.Cm(keepMask), ...
      '.', 'Color',[.55 .55 .55], 'MarkerSize',3); hold on; grid on;
plot3(D.alpha_m(~keepMask), D.beta_m(~keepMask), D.Cm(~keepMask), ...
      'r.', 'MarkerSize',8);
xlabel('\alpha_m [rad]'); ylabel('\beta_m [rad]'); zlabel('C_m');
view(35, 22);
legend('clean samples','flagged (glitch bursts)','Location','best');
title('3-D scatter (\alpha_m, \beta_m, C_m) - glitched samples shown in red');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p11_3d_glitch.png'), 'Resolution', 200);

% 3-D scatter, top-down: structure of the glitch in the (alpha,beta) plane
figure('Name','P1.1 3D top-down view, glitch highlighted','NumberTitle','off');
plot3(D.alpha_m(keepMask),  D.beta_m(keepMask),  D.Cm(keepMask), ...
      '.', 'Color',[.55 .55 .55], 'MarkerSize',3); hold on; grid on;
plot3(D.alpha_m(~keepMask), D.beta_m(~keepMask), D.Cm(~keepMask), ...
      'r.', 'MarkerSize',8);
xlabel('\alpha_m [rad]'); ylabel('\beta_m [rad]');
view(0, 90);   % top-down: look along -C_m
axis equal;
legend('clean samples','flagged (glitch bursts)','Location','best');
title('Top-down (\alpha_m, \beta_m) - glitch stripe is structured, not random');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p11_3d_topdown.png'), 'Resolution', 200);

figure('Name','P1.1 alpha-beta scatter','NumberTitle','off');
plot(D.alpha_m, D.beta_m, '.', 'MarkerSize', 3); grid on;
xlabel('\alpha_m [rad]'); ylabel('\beta_m [rad]');
title('SysID excitation pattern (\alpha_m, \beta_m)');
axis equal;
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p11_spiral.png'), 'Resolution', 200);

fprintf('\n');

%% POINT 2 System and observation equations
% Equations are documented in:

fprintf('1.2  State / observation eqs documented in +kf/*.m\n\n');

%% POINT 3 Filter choice
% • State equation f is LINEAR in x (Fx = 0).

fprintf('1.3  Filter choice: IEKF (h is strongly nonlinear).\n\n');

%% POINT 4 IEKF run + convergence proof
% x0 = [V_m(0); 0; V_m(0)*tan(alpha_m(0)); 0] (point 2)

fprintf('1.4  Running IEKF on %d samples...\n', D.N);

Z = [D.alpha_m, D.beta_m, D.V_m];
U = D.U_k;

params = struct();
params.x0      = [D.V_m(1); 0; D.V_m(1)*tan(D.alpha_m(1)); 0];
params.P0      = diag([1e-2 1e-2 1e-2 1e0]);
params.Q       = diag([1e-6 1e-6 1e-6 0]);
params.R       = diag([2.25e-6 2.25e-6 1]);
params.maxIter = 20;
params.epsTol  = 1e-10;
params.integrator = 'rk4';      % Appendix

tStart = tic;
kfOut  = kf.iekf(Z, U, D.dt, params);
tRk4 = toc(tStart);
fprintf('    IEKF (RK4)   done in %.2f s. Mean inner iters = %.2f, max = %d.\n', ...
        tRk4, mean(kfOut.iters), max(kfOut.iters));

% RK4 vs Euler integrator comparison
paramsE            = params;
paramsE.integrator = 'euler';

tStart  = tic;
kfOutE  = kf.iekf(Z, U, D.dt, paramsE);
tEuler  = toc(tStart);
dx_max  = max(abs(kfOut.xhat(:)  - kfOutE.xhat(:)));
dP_max  = max(abs(kfOut.Pdiag(:) - kfOutE.Pdiag(:)));
dC_max  = max(abs(kfOut.xhat(:,4) - kfOutE.xhat(:,4)));
fprintf('    IEKF (Euler) done in %.2f s.\n', tEuler);
fprintf('    Integrator equivalence check:\n');
fprintf('      max|x_RK4 - x_Euler|            = %.3e\n', dx_max);
fprintf('      max|Pdiag_RK4 - Pdiag_Euler|    = %.3e\n', dP_max);
fprintf('      max|C_RK4(k) - C_Euler(k)|      = %.3e\n', dC_max);
if dx_max < 1e-12
    fprintf('      OK: integrators agree to machine precision (as predicted).\n');
else
    fprintf('      WARN: integrators disagree by more than 1e-12. Re-check.\n');
end

% Tolerance justification print (for report)
fprintf('    Inner-loop tolerance justification (epsTol=%.0e):\n', params.epsTol);
fprintf('      sigma_alpha = 1.5e-3 -> epsTol is %.0f orders below noise level\n', ...
        log10(1.5e-3 / params.epsTol));
fprintf('      |C| ~ O(0.1)         -> epsTol is %.0f orders below bias magnitude\n', ...
        log10(0.1 / params.epsTol));

% Plot: RK4 vs Euler trajectories on the C_alpha_up channel
figure('Name','P1.4 RK4 vs Euler equivalence','NumberTitle','off');
subplot(2,1,1);
plot(D.t, kfOut.xhat(:,4),  '-',  'LineWidth',1); hold on; grid on;
plot(D.t, kfOutE.xhat(:,4), '--', 'LineWidth',1);
xlabel('time [s]'); ylabel('Chat_{\alpha_{up}}');
legend('RK4 integrator','Euler integrator','Location','best');
title(sprintf('Chat_{\\alpha_{up}}(t): RK4 vs Euler   (max|\\Delta|=%.1e)', dC_max));
subplot(2,1,2);
semilogy(D.t, max(abs(kfOut.xhat - kfOutE.xhat), [], 2)+eps); grid on;
xlabel('time [s]'); ylabel('max_j |x_{RK4,j} - x_{Euler,j}|');
title('State-trajectory difference across the IEKF run (floor = eps)');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p14_rk4_vs_euler.png'), 'Resolution', 200);

% Convergence proof plot: Chat_alpha_up(k) and P_kk(4,4)
figure('Name','P1.4 IEKF convergence','NumberTitle','off');
subplot(2,1,1);
plot(D.t, kfOut.xhat(:,4)); grid on;
xlabel('time [s]'); ylabel('Chat_{\alpha_{up}}');
title('Upwash bias estimate (should asymptote to a constant)');
subplot(2,1,2);
semilogy(D.t, kfOut.Pdiag(:,4)); grid on;
xlabel('time [s]'); ylabel('P_{k|k}(4,4)');
title('Posterior variance of C_{\alpha_{up}} (should decrease monotonically)');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p14_Cupwash_conv.png'), 'Resolution', 200);

% Innovation / residual diagnostics.
figure('Name','P1.4 innovation and residual diagnostics','NumberTitle','off');
for j = 1:3
    subplot(3,1,j);
    plot(D.t, kfOut.innov(:,j), 'LineWidth', 1); hold on;
    plot(D.t, kfOut.resid(:,j), '--', 'LineWidth', 1);
    grid on;
    ylabel(sprintf('ch %d', j));
    if j == 1
        legend('a priori innovation','post-fit residual','Location','best');
    end
    if j == 3, xlabel('time [s]'); end
end
sgtitle('IEKF diagnostics: a priori innovation and post-fit residual');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p14_innov.png'), 'Resolution', 200);

% One-number summary for the converged C estimate.
% Using the last part of the run, not only the final sample, so the printed
% value is not decided by one noisy IEKF update. It sometimes gave issues
% otherwise.
nTail   = round(0.10 * D.N);
Cfinal  = mean(kfOut.xhat(end-nTail+1:end, 4));
P44_end = mean(kfOut.Pdiag(end-nTail+1:end, 4));
fprintf('    Chat_alpha_up final (mean over last 10%%):  %.5f\n', Cfinal);
fprintf('    P_kk(4,4)  final:                    %.3e\n', P44_end);

% Gold-standard convergence check against the true state file
stateFile = fullfile(projectRoot, 'F16traindata_CMabV_2026_state.mat');
if isfile(stateFile)
    St = load(stateFile);
    sf = fieldnames(St);
    Xtrue = St.(sf{find(structfun(@(v) size(v,2)>=4, St), 1)});  % the N×4 state array
    Cup_true = Xtrue(1, 4);                       % true C_alpha_up (constant column)
    relErr   = abs(Cfinal - Cup_true) / abs(Cup_true);
    fprintf('    TRUE C_alpha_up (state file):            %.5f\n', Cup_true);
    fprintf('    IEKF recovery error:                 %.2f%% (|Chat-C_true|/C_true)\n', 100*relErr);
else
    fprintf('    (state file not found - truth check skipped)\n');
end

%% POINT 5 - Reconstruct alpha_true
% alpha_true^(A)(k) = alpha_m(k) / (1 + Chat_alfa_up(k)) (analytic inversion)

fprintf('\n1.5  Reconstructing alpha_true...\n');

reconOut = kf.reconAlpha(D.alpha_m, kfOut);
alpha_true = reconOut.alpha_filter;   % the cleaner of the two; used downstream

figure('Name','P1.5 alpha reconstruction','NumberTitle','off');
plot(D.t, D.alpha_m,             '-', 'Color',[.6 .6 .6]); hold on; grid on;
plot(D.t, reconOut.alpha_analytic,'-', 'LineWidth', 1);
plot(D.t, reconOut.alpha_filter, '-', 'LineWidth', 1);
xlabel('time [s]'); ylabel('\alpha [rad]');
legend('\alpha_m (measured, biased + noisy)', ...
       '\alpha_m / (1+Chat)  (analytic inversion)', ...
       'arctan(what/û)       (filter state)', ...
       'Location','best');
title('\alpha reconstruction: analytic vs. filter');
% Clip to the physical alpha range. The analytic inversion alpha_m/(1+Chat) spikes
ylim([-0.4 1.2]);
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p15_alpha_recon.png'), 'Resolution', 200);

%% POINT 6 OLS polynomial fit + degree sweep
% Drop V from the regressor (the velocity dependence of C_m in this

fprintf('\n1.6  OLS polynomial degree sweep...\n');

% Build the Cm-modeling dataset: use reconstructed alpha_true, and apply
alpha_model = alpha_true(keepMask);
beta_model  = D.beta_m(keepMask);
V_model     = D.V_m(keepMask);
Cm_model    = D.Cm(keepMask);

split = data.splitTrainVal(numel(Cm_model), ...
                           struct('method','chunk','valFrac',0.25,'numChunks',8));
fprintf('    %s\n', split.summary);

% V dependence check: directly compare Cm(alpha,beta) vs Cm(alpha,beta,V)
Vrel = std(V_model) / mean(V_model);
fprintf('    V-spread check: std(V_m)/mean(V_m) = %.3g\n', Vrel);
dV       = 5;
splitV   = data.splitTrainVal(numel(Cm_model), ...
                              struct('method','chunk','valFrac',0.25,'numChunks',8));
Xid_2d   = [alpha_model(splitV.idMask), beta_model(splitV.idMask)];
Yid_v    = Cm_model(splitV.idMask);
Xval_2d  = [alpha_model(splitV.valMask),beta_model(splitV.valMask)];
Yval_v   = Cm_model(splitV.valMask);
Xid_3d   = [Xid_2d,  V_model(splitV.idMask)];
Xval_3d  = [Xval_2d, V_model(splitV.valMask)];

A2id  = olspoly.buildA(Xid_2d,  dV);  A2val = olspoly.buildA(Xval_2d, dV);
A3id  = olspoly.buildA(Xid_3d,  dV);  A3val = olspoly.buildA(Xval_3d, dV);
fit2  = olspoly.fitOLS(A2id, Yid_v);
fit3  = olspoly.fitOLS(A3id, Yid_v);
rms2  = sqrt(mean((Yval_v - A2val * fit2.theta).^2));
rms3  = sqrt(mean((Yval_v - A3val * fit3.theta).^2));
% Parameter significance of the V-bearing terms (those whose exponent in V > 0)
[~, expList3] = olspoly.buildA(Xid_3d(1,:), dV);
pc3   = validation.paramCov(A3id, Yid_v - A3id*fit3.theta, struct('theta', fit3.theta));
maskV = expList3(:,3) > 0;
nV    = sum(maskV);
nVins = sum(maskV & pc3.flagged(:));   % V-terms flagged as noise (|t|<2)
fprintf('    At d=%d: in-domain val RMS  (alpha,beta)         = %.4g\n', dV, rms2);
fprintf('    At d=%d: in-domain val RMS  (alpha,beta,V)       = %.4g\n', dV, rms3);
fprintf('    -> RMS change from adding V                       = %+.2e (%+.1f%%)\n', ...
        rms3-rms2, 100*(rms3-rms2)/rms2);
fprintf('    V-bearing terms: %d / %d flagged |t|<2 (noise-level).\n', nVins, nV);
fprintf('    => Drop V (no operational benefit; extra %d params not significant).\n', nV);

% Build the regressor input (alpha_true reconstructed, beta_m original)
Xall = [alpha_model, beta_model];         % N×2
Yall = Cm_model;                          % N×1

Xid  = Xall(split.idMask, :);  Yid  = Yall(split.idMask);
Xval = Xall(split.valMask,:);  Yval = Yall(split.valMask);

% Load the 100-point special-validation grid HERE (not only in 1.8), because
S = data.loadSpecial(projectRoot);

degSweep     = 1:10;
RMS_id       = zeros(size(degSweep));
RMS_val      = zeros(size(degSweep));
condAtA      = zeros(size(degSweep));
RMS_spec_all = zeros(size(degSweep));   % special grid, all 100 points
RMS_spec_in  = zeros(size(degSweep));   % special grid, inside training hull
RMS_spec_out = zeros(size(degSweep));   % special grid, outside hull (extrapolation)
polyOuts     = cell(size(degSweep));

for k = 1:numel(degSweep)
    d_k = degSweep(k);
    A_id  = olspoly.buildA(Xid,  d_k);
    A_val = olspoly.buildA(Xval, d_k);
    polyOuts{k} = olspoly.fitOLS(A_id, Yid);
    RMS_id(k)  = polyOuts{k}.RMS;
    RMS_val(k) = sqrt( mean( (Yval - A_val * polyOuts{k}.theta).^2 ) );
    condAtA(k) = polyOuts{k}.condAtA;

    % Item 7's second half: evaluate the SAME fit on the special grid,
    specSweep = validation.specialRMS( ...
        @(Xq) olspoly.buildA(Xq, d_k) * polyOuts{k}.theta, ...
        S.alpha_val, S.beta_val, S.Cm_val, Xid);
    RMS_spec_all(k) = specSweep.RMS_all;
    RMS_spec_in(k)  = specSweep.RMS_in;
    RMS_spec_out(k) = specSweep.RMS_out;
end

% Order selection: in-domain diagnostics; SPECIAL GRID SEALED
parsimonyTol = 0.20;                       % parsimony band
validMask    = condAtA < 1/eps;            % conditioning-valid degrees (-> d<=7)
bestValid    = min(RMS_val(validMask));    % best in-domain val among valid d
selMask      = validMask & (RMS_val <= (1 + parsimonyTol) * bestValid);
kStar        = find(selMask, 1, 'first');  % smallest qualifying degree (parsimony)
dStar        = degSweep(kStar);
[~, kv]      = min(RMS_val);               % raw argmin (fallacy), for contrast
fprintf(['    Order selection (in-domain only; special grid for 1.9):\n' ...
         '      raw argmin in-domain RMS_val   -> d%d  (on-arm: blind to off-arm Runge)\n' ...
         '      conditioning-valid range       -> d <= %d  (cond(A''A) < 1/eps)\n' ...
         '      parsimony (within %.0f%% of best) -> d%d  (CHOSEN)\n'], ...
         degSweep(kv), max(degSweep(validMask)), 100*parsimonyTol, dStar);
fprintf('    Chosen d* = %d  (RMS_val = %.4g, kappa(A''A) = %.2e < 1/eps).\n', ...
         dStar, RMS_val(kStar), condAtA(kStar));

% Plot order-vs-accuracy on BOTH datasets + condition number
figure('Name','P1.7 OLS poly degree sweep','NumberTitle','off');
subplot(2,1,1);
h1 = semilogy(degSweep, RMS_id,       '-o', 'LineWidth',1); hold on; grid on;
h2 = semilogy(degSweep, RMS_val,      '-s', 'LineWidth',1);
h3 = semilogy(degSweep, RMS_spec_in,  '-^', 'LineWidth',1);
h4 = semilogy(degSweep, RMS_spec_out, '-v', 'LineWidth',1);
h5 = semilogy(degSweep, RMS_spec_all, '-d', 'LineWidth',2.4, 'Color',[0 0 0]);
xlabel('degree d'); ylabel('RMS');
legend([h1 h2 h3 h4 h5], ...
       'id', 'validation', 'special inside hull', 'special outside hull', ...
       'special total', ...
       'Location','southoutside', 'NumColumns', 3);
title('Accuracy vs. polynomial order');
xline(dStar, '--k', 'HandleVisibility','off');   % keep out of the legend
plot(dStar, RMS_val(kStar), 'kp', 'MarkerFaceColor','y', ...
     'MarkerSize',12, 'HandleVisibility','off');
text(dStar, RMS_val(kStar), sprintf('  d^*=%d (in-domain parsimony)',dStar), ...
     'VerticalAlignment','top');
% kappa(A)=sqrt(kappa(A'A)) (what the QR solve `A\Y` that produces theta
subplot(2,1,2);
semilogy(degSweep, condAtA,       '-o', 'LineWidth',1.2); hold on; grid on;
semilogy(degSweep, sqrt(condAtA), '-s', 'LineWidth',1.2);
yline(1/eps, ':', '1/eps \approx 4.5\times10^{15}', 'HandleVisibility','off', ...
      'LabelHorizontalAlignment','left');
xlabel('degree d'); ylabel('condition number');
legend('\kappa(A^TA) - normal-eqn form, covariance & t-stats', ...
       '\kappa(A)=\surd\kappa(A^TA) - what the QR coeff-solve sees', ...
       'Location','southoutside', 'NumColumns', 1);
title('Conditioning of the polynomial regression matrix');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p17_degsweep.png'), 'Resolution', 200);

% Prediction surfaces vs. degree: OVERALL behaviour + in-hull zoom
degShow = [2 5 6 8 10];   % underfit \dot d* (chosen) \dot plateau \dot diverging \dot catastrophic
nShow   = numel(degShow);

% Operational grid = bounding box of (training data U special-val grid).
aLo = min([alpha_model; S.alpha_val]);  aHi = max([alpha_model; S.alpha_val]);
bLo = min([beta_model;  S.beta_val ]);  bHi = max([beta_model;  S.beta_val ]);
[AA, BB] = meshgrid(linspace(aLo,aHi,120), linspace(bLo,bHi,120));
Kh     = convhull(alpha_model, beta_model);
inHull = inpolygon(AA, BB, alpha_model(Kh), beta_model(Kh));

cLo = min(Cm_model);  cHi = max(Cm_model);     % data range
zTop = [cLo-0.35, cHi+0.22];                   % wide: reveal out-of-hull divergence
zBot = [cLo-0.03, cHi+0.03];                   % tight: in-hull shape detail

figure('Name','P1.7 prediction surfaces','NumberTitle','off', ...
       'Position',[20 20 1900 980]);
for ii = 1:nShow
    d_show  = degShow(ii);
    Cm_full = reshape(olspoly.buildA([AA(:), BB(:)], d_show) * polyOuts{d_show}.theta, size(AA));

    % TOP: full operational domain, NO blanking. Values are clamped to zTop
    subplot(2, nShow, ii);
    surf(AA, BB, min(max(Cm_full, zTop(1)), zTop(2)), 'EdgeColor','none'); hold on;
    plot3(alpha_model(Kh), beta_model(Kh), zTop(1)*ones(numel(Kh),1), ...
          'r-', 'LineWidth',1.3);                % training-hull footprint
    xlabel('\alpha_{true}'); ylabel('\beta_m'); zlabel('C_m');
    title(sprintf('d=%d, outside %.2g', d_show, RMS_spec_out(d_show)));
    clim(zBot); zlim(zTop); view(40,25); grid on; box on;

    % BOTTOM: zoom inside the training hull (blanked), tight scale.
    Cm_in = Cm_full; Cm_in(~inHull) = NaN;
    subplot(2, nShow, nShow+ii);
    surf(AA, BB, Cm_in, 'EdgeColor','none'); hold on;
    scatter3(alpha_model(1:8:end), beta_model(1:8:end), Cm_model(1:8:end), ...
             2, [.6 .6 .6], 'filled');
    xlabel('\alpha_{true}'); ylabel('\beta_m'); zlabel('C_m');
    title(sprintf('d=%d, val %.1e', d_show, RMS_val(d_show)));
    clim(zBot); zlim(zBot); view(40,25); grid on; box on;
end
sgtitle('OLS polynomial surfaces: full grid above, training-hull zoom below');
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p17b_pred_surfaces.png'), 'Resolution', 200);

%% POINT 7 Model + statistical validation of d*
% Residual time series (whiteness)

fprintf('\n1.7  Validating d* = %d...\n', dStar);

bestFit = polyOuts{kStar};
A_id    = olspoly.buildA(Xid, dStar);
A_val   = olspoly.buildA(Xval, dStar);

% Held-out diagnostics: evaluate model quality on unseen validation rows.
Yhat_val = A_val * bestFit.theta;
resOut   = validation.residuals(Yval, Yhat_val, Xval);
acOut    = validation.autocorr(resOut.eps);

% Parameter significance diagnostics must use the identification design
resOut_id = validation.residuals(Yid, bestFit.Yhat, Xid);
pcOut     = validation.paramCov(A_id, resOut_id.eps, ...
                                struct('theta', bestFit.theta));

fprintf('    Residual RMS:      %.4g\n', resOut.RMS);
fprintf('    Residual mean:     %.3g  (should be ~0)\n', resOut.mean);
fprintf('    Residual skew/kurt: %.3g / %.3g  (should be ~0)\n', ...
        resOut.skewness, resOut.kurtosis);
fprintf('    Autocorr in 95%% band: %.1f%%  (should be ≥95)\n', acOut.pctInBand);
fprintf('    # noise-level params (|t|<2): %d / %d\n', ...
        sum(pcOut.flagged), numel(pcOut.flagged));

% Residual plots
figure('Name','P1.8 residual diagnostics','NumberTitle','off');
subplot(2,2,1);
plot(resOut.eps); grid on; title('residual time series'); xlabel('k');
subplot(2,2,2);
histogram(resOut.eps, resOut.histEdges); grid on; title('residual histogram');
subplot(2,2,3);
scatter3(Xval(:,1), Xval(:,2), resOut.eps, 4, resOut.eps, 'filled'); grid on;
view(2); colorbar; title('residual surface over (alpha,β)');
xlabel('\alpha_{true}'); ylabel('\beta_m');
subplot(2,2,4);
stem(acOut.lags, acOut.R, 'filled'); hold on; grid on;
yline(acOut.band, '--r'); yline(-acOut.band, '--r');
title(sprintf('autocorrelation (%.0f%% in 95%% band)', acOut.pctInBand));
xlabel('lag \tau'); ylabel('R(\tau)');
sgtitle(sprintf('OLS polynomial d*=%d - validation diagnostics', dStar));
figstyle(gcf); exportgraphics(gcf, fullfile(figDir, 'fig_p18_diag.png'), 'Resolution', 200);

%% POINT 8 Special-validation RMS
% The 100-point special grid extends past the training hull. Report

fprintf('\n1.8  Special-validation RMS...\n');

% S (the special grid) is already loaded in 1.6 for the order sweep; reuse it.
predFun_poly = @(Xq) olspoly.buildA(Xq, dStar) * bestFit.theta;
specOut_poly = validation.specialRMS(predFun_poly, S.alpha_val, S.beta_val, ...
                                      S.Cm_val, Xid);
fprintf('    %s\n', specOut_poly.summary);

%% Save outputs for downstream parts

% recon.mat - bundle reconstructed + GLITCH-CLEANED data (Parts 2/3
recon.alpha_true = alpha_model;   % glitch-cleaned, 8644 long
recon.beta_m     = beta_model;    % glitch-cleaned, 8644 long
recon.V_m        = V_model;       % glitch-cleaned, 8644 long
recon.Cm         = Cm_model;      % glitch-cleaned, 8644 long
recon.t          = D.t(keepMask); % glitch-cleaned, 8644 long
recon.idMask     = split.idMask;
recon.valMask    = split.valMask;
recon.Cfinal     = Cfinal;
recon.dt         = D.dt;
% Bookkeeping, I need to also expose the raw (10001-long) signals + keepMask so
recon.alpha_true_raw = alpha_true;
recon.beta_m_raw     = D.beta_m;
recon.V_m_raw        = D.V_m;
recon.Cm_raw         = D.Cm;
recon.t_raw          = D.t;
recon.keepMask       = keepMask;
save(fullfile(codeRoot, 'recon.mat'), '-struct', 'recon');
fprintf('\nSaved recon.mat (used by run_part2 / run_part3).\n');
fprintf('    Cleaned arrays length = %d (8644 = 10001 - 1357 glitch samples).\n', ...
        numel(recon.Cm));
fprintf('    idMask / valMask align 1-to-1 with the cleaned arrays.\n');

% p1_poly.mat is the chosen polynomial fit + sweep
p1_poly.dStar      = dStar;
p1_poly.theta      = bestFit.theta;
p1_poly.expList    = bestFit.theta * 0;  % placeholder to initialize
[~, p1_poly.expList] = olspoly.buildA(Xid(1,:), dStar);
p1_poly.RMS_id       = RMS_id;
p1_poly.RMS_val      = RMS_val;
p1_poly.RMS_spec_all = RMS_spec_all;   % special grid vs degree (Item 7)
p1_poly.RMS_spec_in  = RMS_spec_in;
p1_poly.RMS_spec_out = RMS_spec_out;
p1_poly.specOut      = specOut_poly;
p1_poly.condAtA      = condAtA;
p1_poly.resOut     = resOut;
p1_poly.acOut      = acOut;
p1_poly.pcOut      = pcOut;
save(fullfile(codeRoot, 'p1_poly.mat'), '-struct', 'p1_poly');
fprintf('Saved p1_poly.mat (used by run_part4).\n');

fprintf('\n===== Part 1 complete =====\n');
