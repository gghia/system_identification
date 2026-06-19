function [bary, simplexIdx] = cart2bary(V, T, X)
% Inputs: V and T define the simplex mesh, X are query points.
% Outputs: bary gives barycentric coordinates, simplexIdx tells which simplex was used.
% CART2BARY Cartesian -> barycentric conversion with simplex membership.

    % Input checking
    if isempty(V) || size(V, 2) < 1
        error('simplex:cart2bary:badV', ...
              'V must be Nv x n with at least 1 column.');
    end

    n = size(V, 2);                          % spatial dimension (2 for 2D)
    if size(X, 2) ~= n
        error('simplex:cart2bary:dimMismatch', ...
              'X has %d columns but V has %d. They must match.', ...
              size(X, 2), n);
    end

    % Default T = [1 2 ... n+1] for the trivial single-simplex case.
    if isempty(T)
        T = 1 : (n + 1);
    end
    if size(T, 2) ~= n + 1
        error('simplex:cart2bary:badT', ...
              'T must be Nt x %d (got %d columns).', n + 1, size(T, 2));
    end

    Nx = size(X, 1);
    Nt = size(T, 1);

    % Dispatch on triangulation size
    if Nt == 1
        % SINGLE-SIMPLEX PATH (Part 2): use the supplied routine directly.
        simplexVerts = V(T(1, :), :);                    % (n+1) x n
        bary = bsplinen_cart2bary(simplexVerts, X);      % Nx x (n+1)

        % Flag out-of-hull points (any b_i < 0) per eq. (3). We keep their
        simplexIdx = ones(Nx, 1);
        outMask = any(bary < -eps('single'), 2);   % tolerant tiny-negative
        simplexIdx(outMask) = NaN;
    else
        % MULTI-SIMPLEX PATH (Part 3): tsearchn does both jobs at once.
        [simplexIdx, bary] = tsearchn(V, T, X);

        % Sanity: tsearchn returns bary with (n+1) columns.
        if size(bary, 2) ~= n + 1
            error('simplex:cart2bary:tsearchnShape', ...
                  ['tsearchn returned bary with %d columns; expected %d. ', ...
                   'Check V/T dimensions.'], size(bary, 2), n + 1);
        end
    end

    % Final partition-of-unity sanity
    rowsOK = ~isnan(simplexIdx);
    if any(rowsOK)
        s = sum(bary(rowsOK, :), 2);
        if any(abs(s - 1) > 1e-10)
            warning('simplex:cart2bary:partitionUnity', ...
                    ['Barycentric rows do not sum to 1 within 1e-10 ', ...
                     '(worst |s-1| = %.3g). Check V/T consistency.'], ...
                    max(abs(s - 1)));
        end
    end
end
