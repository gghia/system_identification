function [kappa, multinomCoef] = multiIndex(n, d)
% Inputs: n is dimension, d is the B-form polynomial degree.
% Outputs: kappa is the exponent list, multinomCoef is d!/k! for each row.
% MULTIINDEX Exponent list for the degree-d B-form basis.
%
% Each row of kappa says which powers of the barycentric coordinates are
% used by one basis function. For a triangle n = 2, so one row has three
% entries [k0 k1 k2], and they must add up to d.
%
% bcoefBasis uses the same rows in:
%   B_k(b) = (d!/k!) * b0^k0 * b1^k1 * ... * bn^kn
% The row order also becomes the column order of B and the order of the
% fitted coefficients c, so this convention must stay fixed everywhere.

    % First list all possible exponent rows with sum(k) = d.
    raw = compositionsOfD(d, n + 1);    % one row per B-form basis function

    % The largest first coordinate first, then largest second
    % coordinate, etc to match the B-net / coefficient ordering
    kappa = sortrows(raw, 1:(n+1), 'descend');

    % cheap check
    assert(all(sum(kappa, 2) == d), 'multiIndex: row sums must equal d (bug).');

    % The basis formula needs d!/k! next to each row.
    factD = factorial(d);
    multinomCoef = factD ./ prod(factorial(kappa), 2);
end

% Recursive list builder for all nonnegative rows with sum(row) = d.
% Here k is the number of slots still to fill.
% Example: compositionsOfD(3, 2) returns [3 0; 2 1; 1 2; 0 3].
function C = compositionsOfD(d, k)
% Inputs: d is the target sum, k is the number of slots left to fill.
% Outputs: C lists all nonnegative rows whose entries sum to d.
    if k == 1
        C = d;
        return;
    end
    rows = nchoosek(d + k - 1, k - 1);
    C    = zeros(rows, k);
    row  = 0;
    for first = d : -1 : 0
        sub = compositionsOfD(d - first, k - 1);
        m   = size(sub, 1);
        C(row+1 : row+m, 1)     = first;
        C(row+1 : row+m, 2:end) = sub;
        row = row + m;
    end
end
