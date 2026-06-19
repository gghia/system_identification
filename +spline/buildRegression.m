function [Bglobal, info] = buildRegression(X, V, T, d)
% Inputs: X are sample points, V/T define the mesh, d is the B-form degree.
% Outputs: Bglobal is the sparse regression matrix, info stores indexing data.
% BUILDREGRESSION Global block-sparse regression matrix B 

    % Sizes & sanity checks
    [N, n]    = size(X);
    nT        = size(T, 1);
    if size(T, 2) ~= n + 1
        error('spline:buildRegression:dim', ...
              'T has %d cols; expected n+1 = %d (n inferred from X).', ...
              size(T,2), n+1);
    end
    if size(V, 2) ~= n
        error('spline:buildRegression:dim', ...
              'V has %d cols; expected n = %d to match X.', size(V,2), n);
    end

    % One-time basis bookkeeping
    [kappa, multinomCoef] = simplex.multiIndex(n, d);
    dHat = size(kappa, 1);          % = nchoosek(d + n, n)

    % To locate every data point: simplex + barycentric coords
    [simplexIdx, barys] = tsearchn(V, T, X);

    inHullMask = ~isnan(simplexIdx);
    nIn        = nnz(inHullMask);

    % Build sparse triplets
    rowIdx = zeros(nIn * dHat, 1);
    colIdx = zeros(nIn * dHat, 1);
    vals   = zeros(nIn * dHat, 1);

    % We iterate over IN-HULL data points only. The "new row index" of
    inIdx = find(inHullMask);          % nIn × 1, original positions
    rowOut = (1:nIn).';                % nIn × 1, output row of Bglobal

    % Evaluate the basis row-by-row, could be vectorized
    ptr = 0;
    for ii = 1:nIn
        i_orig = inIdx(ii);
        j      = simplexIdx(i_orig);          % simplex index for this point
        b_row  = barys(i_orig, :);            % 1 × (n+1) barycentric
        % B-form basis row (1 × d̂) at this point:
        Brow   = simplex.bcoefBasis(b_row, kappa, multinomCoef);

        % Place into columns (j-1)*dHat + 1 : j*dHat of OUTPUT row rowOut(ii).
        colRange = (j - 1) * dHat + (1:dHat);
        slots    = ptr + (1:dHat);
        rowIdx(slots) = rowOut(ii);
        colIdx(slots) = colRange.';
        vals(slots)   = Brow.';
        ptr = ptr + dHat;
    end

    Bglobal = sparse(rowIdx, colIdx, vals, nIn, nT * dHat);

    % Pack info
    info.simplexIdx   = simplexIdx;
    info.inHullMask   = inHullMask;
    info.dHat         = dHat;
    info.T            = nT;
    info.kappa        = kappa;
    info.multinomCoef = multinomCoef;
    info.n            = n;
    info.barys        = barys;
end
