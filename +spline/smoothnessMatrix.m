function [H, info] = smoothnessMatrix(V, T, d, r)
% Inputs: V/T define the mesh, d is spline degree, r is requested continuity.
% Outputs: H is the sparse constraint matrix, info stores edge/bookkeeping data.
% SMOOTHNESSMATRIX Global smoothness matrix H for a 2D spline mesh.

    % Basic sanity
    n = size(V, 2);
    if size(T, 2) ~= n + 1
        error('spline:smoothnessMatrix:dim', ...
              'T has %d cols; expected n+1 = %d.', size(T,2), n+1);
    end
    if n ~= 2
        % This code is written for triangle meshes. Some size checks are
        % generic, but the edge-orientation helper below is only 2D.
        error('spline:smoothnessMatrix:dim', ...
              ['This implementation is specialised to n = 2 (2D triangles); ' ...
               'got n = %d. Generalise the local helper permuteToEdgeCanonical ' ...
               'to tetrahedra (n = 3) if you need 3D.'], n);
    end
    if r < 0
        error('spline:smoothnessMatrix:r', 'r must be >= 0; got %g.', r);
    end
    if r > d - 1
        warning('spline:smoothnessMatrix:r', ...
                'r = %d but d = %d; r should be < d. Continuing - H will likely be over-determined.', r, d);
    end

    nT = size(T, 1);

    % Multi-index list and fast row-to-column lookup.
    [kappa, ~] = simplex.multiIndex(n, d);
    dHat       = size(kappa, 1);                     % = nchoosek(d + n, n)

    % The smoothness rows need to jump from an exponent row to its column.
    % A text key is a simple way to do that lookup quickly.
    kappaIndex = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    for l = 1:dHat
        kappaIndex(encodeKappa(kappa(l, :))) = int32(l);
    end

    % Enumerate interior edges
    edgeRows = zeros(3 * nT, 3);          % [vMin, vMax, simplexIdx]
    e = 0;
    for jj = 1:nT
        verts = T(jj, :);
        % Three edges of triangle jj: vertex pairs (1,2), (1,3), (2,3).
        pairs = nchoosek(verts, 2);       % 3 x 2
        pairs = sort(pairs, 2);           % canonical (min, max)
        for kk = 1:3
            e = e + 1;
            edgeRows(e, :) = [pairs(kk, :), jj];
        end
    end
    edgeRows = edgeRows(1:e, :);

    % Group by (vMin, vMax). Interior edges appear in 2 rows.
    [~, ~, gid] = unique(edgeRows(:, 1:2), 'rows');
    edges = [];                            % E x 4: [t1, t2, vA, vB]
    for g = 1:max(gid)
        rows = find(gid == g);
        if numel(rows) == 2
            % Interior edge shared by two simplices.
            t1 = edgeRows(rows(1), 3);
            t2 = edgeRows(rows(2), 3);
            vA = edgeRows(rows(1), 1);
            vB = edgeRows(rows(1), 2);
            edges = [edges; t1, t2, vA, vB]; %#ok<AGROW>
        elseif numel(rows) > 2
            % An edge shared by more than two triangles is not a valid mesh.
            error('spline:smoothnessMatrix:manifold', ...
                  'Edge (%d, %d) is shared by %d simplices; expected 2.', ...
                  edgeRows(rows(1),1), edgeRows(rows(1),2), numel(rows));
        end
        % numel(rows) == 1 means a BOUNDARY edge - no continuity to enforce.
    end

    E = size(edges, 1);

    % Row budget
    R = zeros(r + 1, 1);
    for m = 0:r
        R(m + 1) = nchoosek(d - m + n - 1, n - 1);
    end
    rowsPerEdge = sum(R);
    nRows       = E * rowsPerEdge;

    % Build H as sparse triplets
    maxRowNnz = 1 + max(arrayfun(@(m) nchoosek(m + n, n), 0:r));
    nnzMax    = nRows * maxRowNnz;
    rowIdx = zeros(nnzMax, 1);
    colIdx = zeros(nnzMax, 1);
    vals   = zeros(nnzMax, 1);
    ptr    = 0;
    rowNo  = 0;

    for ee = 1:E
        t1 = edges(ee, 1);
        t2 = edges(ee, 2);
        vA = edges(ee, 3);     % shared vertex A (global index)
        vB = edges(ee, 4);     % shared vertex B (global index)

        % Find how the shared edge vertices appear in each local triangle.
        verts1 = T(t1, :);
        verts2 = T(t2, :);
        [permKappa1, outVertGlobal1] = permuteToEdgeCanonical(verts1, vA, vB);
        [permKappa2, ~              ] = permuteToEdgeCanonical(verts2, vA, vB);
        % permKappa maps local vertex order into the common edge order:
        % [first edge vertex, opposite vertex, second edge vertex].

        % Opposite vertex of t1, written in barycentric coordinates of t2.
        v_star_1_xy = V(outVertGlobal1, :);                 % 1 x n
        V_t2_rows   = V(T(t2, :), :);                       % (n+1) x n
        b_star_local = bsplinen_cart2bary(V_t2_rows, v_star_1_xy);  % 1 x (n+1)
        % Put those barycentric coordinates in the same common edge order.
        b_star = b_star_local(permKappa2);                  % 1 x (n+1)

        % Build rows for continuity orders m = 0, 1, ..., r.
        for m = 0:r
            % Left side rows have exponent pattern [k0, m, k1].
            for k0 = 0:(d - m)
                k1 = d - m - k0;
                rowNo = rowNo + 1;

                % Canonical left-side exponent row.
                kL_canonical = [k0, m, k1];

                % Convert back to t1 local ordering before looking up the column.
                kL_local_t1 = zeros(1, n + 1);
                kL_local_t1(permKappa1) = kL_canonical;
                colL = (t1 - 1) * dHat + double(kappaIndex(encodeKappa(kL_local_t1)));

                ptr = ptr + 1;
                rowIdx(ptr) = rowNo;
                colIdx(ptr) = colL;
                vals(ptr)   = 1;

                % Right side: add all degree-m basis terms at b_star.
                gammaList   = compositionsOfM(m, n + 1);    % number of terms x (n+1)
                multinomM   = factorial(m) ./ prod(factorial(gammaList), 2);
                % Basis row at b_star for the degree-m terms.
                Bm_at_bstar = bcoefRow(b_star, gammaList, multinomM);   % 1 x number of terms

                for gg = 1:size(gammaList, 1)
                    gamma_canonical = gammaList(gg, :);
                    % t_2 multi-index in canonical orientation:
                    kR_canonical = [k0, 0, k1] + gamma_canonical;
                    % Back to t_2's LOCAL ordering:
                    kR_local_t2 = zeros(1, n + 1);
                    kR_local_t2(permKappa2) = kR_canonical;
                    colR = (t2 - 1) * dHat + ...
                           double(kappaIndex(encodeKappa(kR_local_t2)));

                    ptr = ptr + 1;
                    rowIdx(ptr) = rowNo;
                    colIdx(ptr) = colR;
                    vals(ptr)   = -Bm_at_bstar(gg);
                end
            end
        end
    end

    % Trim and assemble.
    rowIdx = rowIdx(1:ptr);
    colIdx = colIdx(1:ptr);
    vals   = vals(1:ptr);
    H      = sparse(rowIdx, colIdx, vals, nRows, nT * dHat);

    % Pack info
    info.edges       = edges;
    info.numEdges    = E;
    info.rowsPerEdge = rowsPerEdge;
    info.R           = R;
    info.dHat        = dHat;
    info.T           = nT;
    info.kappa       = kappa;
    info.kappaIndex  = kappaIndex;
end

% LOCAL HELPERS

function key = encodeKappa(k)
% Inputs: k is one exponent row.
% Outputs: key is a text key for containers.Map lookup.
% ENCODEKAPPA Hash a multi-index row into a string key for containers.Map.
    key = sprintf('%d,', k);
end

function [perm, outVertGlobal] = permuteToEdgeCanonical(verts, vA, vB)
% Inputs: verts is one triangle, vA/vB are the shared edge vertices.
% Outputs: perm maps canonical edge ordering to local ordering, outVertGlobal is the opposite vertex.
% PERMUTETOEDGECANONICAL Put local triangle vertices in shared-edge order.
    idxA = find(verts == vA, 1);
    idxB = find(verts == vB, 1);
    if isempty(idxA) || isempty(idxB)
        error('spline:smoothnessMatrix:badEdge', ...
              'Edge vertex %d or %d not found in triangle [%d %d %d].', ...
              vA, vB, verts(1), verts(2), verts(3));
    end
    idxStar = setdiff(1:3, [idxA, idxB]);
    perm    = [idxA, idxStar, idxB];   % canonical to local component map
    outVertGlobal = verts(idxStar);
end

function C = compositionsOfM(m, k)
% Inputs: m is the target sum, k is the number of slots.
% Outputs: C lists all nonnegative k-tuples whose rows sum to m.
% COMPOSITIONSOFM All k-tuples of nonnegative integers summing to m.
    if k == 1
        C = m;
        return;
    end
    rows = nchoosek(m + k - 1, k - 1);
    C    = zeros(rows, k);
    row  = 0;
    for first = m : -1 : 0
        sub = compositionsOfM(m - first, k - 1);
        sz  = size(sub, 1);
        C(row+1 : row+sz, 1)     = first;
        C(row+1 : row+sz, 2:end) = sub;
        row = row + sz;
    end
end

function vals = bcoefRow(b, gammaList, multinomM)
% Inputs: b is one barycentric row, gammaList/multinomM define degree-m basis rows.
% Outputs: vals is the row of B-form basis values at b.
% BCOEFROW Row of degree-m B-form basis values at a single bary point.
    G = size(gammaList, 1);
    vals = zeros(1, G);
    for gg = 1:G
        gamma = gammaList(gg, :);
        prodB = 1;
        for kk = 1:numel(gamma)
            if gamma(kk) > 0
                prodB = prodB * (b(kk) ^ gamma(kk));
            end
        end
        vals(gg) = multinomM(gg) * prodB;
    end
end
