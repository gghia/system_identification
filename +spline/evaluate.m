function [Yq, info] = evaluate(Xq, V, T, d, c, opts)
% Inputs: Xq are query points, V/T/d/c define the fitted spline, opts handles hull policy.
% Outputs: Yq are predictions, info stores simplex indices and barycentric coordinates.
% EVALUATE Evaluates a fitted multivariate simplex spline at query points.

    if nargin < 6 || isempty(opts), opts = struct(); end
    if ~isfield(opts, 'outOfHull'), opts.outOfHull = 'nan'; end
    policy = lower(opts.outOfHull);
    if ~ismember(policy, {'nan', 'nearest', 'drop'})
        error('spline:evaluate:opts', ...
              'opts.outOfHull must be ''nan'', ''nearest'', or ''drop''; got ''%s''.', ...
              opts.outOfHull);
    end

    [Nq, n] = size(Xq);
    nT      = size(T, 1);
    if size(T, 2) ~= n + 1
        error('spline:evaluate:dim', 'T has %d cols; expected n+1 = %d.', size(T,2), n+1);
    end

    [kappa, multinomCoef] = simplex.multiIndex(n, d);
    dHat = size(kappa, 1);
    if numel(c) ~= nT * dHat
        error('spline:evaluate:dim', ...
              'c has %d entries; expected nT · d̂ = %d · %d = %d.', ...
              numel(c), nT, dHat, nT * dHat);
    end

    % Locate query points
    [simplexIdx, barys] = tsearchn(V, T, Xq);
    inHullMask          = ~isnan(simplexIdx);

    % Out-of-hull policy
    if strcmp(policy, 'nearest') && any(~inHullMask)
        % Snap each out-of-hull point to the nearest simplex (centroid)
        oohRows = find(~inHullMask);
        % Pre-compute simplex centroids once.
        centroids = zeros(nT, n);
        for j = 1:nT
            centroids(j, :) = mean(V(T(j, :), :), 1);
        end
        for ii = 1:numel(oohRows)
            i = oohRows(ii);
            % Find nearest simplex.
            dists = sum((centroids - Xq(i, :)).^2, 2);
            [~, jNear] = min(dists);
            simplexIdx(i) = jNear;
            % Bary coords of Xq(i,:) w.r.t. jNear
            V_t     = V(T(jNear, :), :);
            barys(i, :) = bsplinen_cart2bary(V_t, Xq(i, :));
        end
        % inHullMask is unchanged, it still marks the original truth
    end

    % Build the evaluation Yq.
    Yq = nan(Nq, 1);

    % Decide which queries to evaluate based on policy.
    switch policy
        case 'nan'
            queriesToEval = find(inHullMask);
        case 'nearest'
            queriesToEval = (1:Nq).';      % all; out-of-hull were snapped
        case 'drop'
            queriesToEval = find(inHullMask);  % evaluate only in-hull
    end

    for kk = 1:numel(queriesToEval)
        i = queriesToEval(kk);
        j = simplexIdx(i);
        % c-block for simplex j: indices (j-1)*dHat + 1 : j*dHat.
        cBlock = c((j - 1) * dHat + (1:dHat));
        Brow   = simplex.bcoefBasis(barys(i, :), kappa, multinomCoef);  % 1 x dhat
        Yq(i)  = Brow * cBlock;
    end

    % If policy is 'drop', compress Yq to in-hull entries only.
    if strcmp(policy, 'drop')
        Yq = Yq(inHullMask);
    end

    % Pack info
    info.simplexIdx = simplexIdx;
    info.inHullMask = inHullMask;
    info.barys      = barys;
    info.policy     = policy;
end
