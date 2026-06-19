function [V, T] = triangulate(X, mode, opts)
% Inputs: X is the point cloud, mode selects the triangulation, opts stores settings.
% Outputs: V are vertices and T are triangle vertex indices.
% TRIANGULATE Builds a 2-D triangulation (V, T) for simplex(-spline) fits.

    if nargin < 3, opts = struct(); end
    if nargin < 2 || isempty(mode), mode = 'bounding'; end

    mode = lower(char(mode));

    switch mode
        case 'bounding'        % single triangle covering all of X
            if ~isfield(opts, 'margin'),  opts.margin = 0.05; end
            if isempty(X) || size(X, 2) ~= 2
                error('simplex:triangulate:badX', ...
                      '''bounding'' mode requires X to be N × 2.');
            end

            % Inflated bounding box of the data.
            xmin = min(X(:, 1));   xmax = max(X(:, 1));
            ymin = min(X(:, 2));   ymax = max(X(:, 2));
            dx   = xmax - xmin;    dy   = ymax - ymin;
            mx   = opts.margin * max(dx, eps);
            my   = opts.margin * max(dy, eps);

            xL = xmin - mx;   xH = xmax + mx;
            yL = ymin - my;   yH = ymax + my;

            % Right-triangle with apex at (xL, yL); long legs reach far
            V = [ xL,            yL;            ...  % v_0  (lower-left apex)
                  xL + 2*(xH-xL), yL;            ...  % v_1  (far right)
                  xL,             yL + 2*(yH-yL) ];  % v_2  (far top)
            T = [1 2 3];

            % to warn if any data point would map to a negative
            
                [b, ~] = simplex.cart2bary(V, T, X);
                if any(b(:) < -1e-10)
                    warning('simplex:triangulate:tightBounding', ...
                            ['Bounding triangle does not strictly contain ', ...
                             'all data (min bary = %.3g). Increase ', ...
                             'opts.margin.'], min(b(:)));
                end
            

        case 'delaunay'        % multi-simplex on a regular vertex grid
            if ~isfield(opts, 'nx'),      opts.nx     = 5;    end
            if ~isfield(opts, 'ny'),      opts.ny     = 5;    end
            if ~isfield(opts, 'margin'),  opts.margin = 0.05; end
            if isempty(X) || size(X, 2) ~= 2
                error('simplex:triangulate:badX', ...
                      '''delaunay'' mode requires X to be N × 2.');
            end
            if opts.nx < 2 || opts.ny < 2
                error('simplex:triangulate:gridTooSmall', ...
                      'opts.nx and opts.ny must both be ≥ 2.');
            end

            % Inflated bounding box, as in 'bounding'.
            xmin = min(X(:, 1));   xmax = max(X(:, 1));
            ymin = min(X(:, 2));   ymax = max(X(:, 2));
            dx   = xmax - xmin;    dy   = ymax - ymin;
            mx   = opts.margin * max(dx, eps);
            my   = opts.margin * max(dy, eps);

            % Regular vertex grid covering the inflated bbox.
            xs = linspace(xmin - mx, xmax + mx, opts.nx);
            ys = linspace(ymin - my, ymax + my, opts.ny);
            [Xg, Yg] = ndgrid(xs, ys);
            V  = [Xg(:), Yg(:)];                     % Nv × 2

            % Delaunay triangulation. delaunayn returns rows of vertex
            T = delaunayn(V);                        % Nt × 3

            % Sanity: T should have 2·(nx-1)·(ny-1) simplices
            Texpected = 2 * (opts.nx - 1) * (opts.ny - 1);
            if size(T, 1) ~= Texpected
                warning('simplex:triangulate:unexpectedT', ...
                        ['delaunayn returned %d simplices; expected %d ', ...
                         'for a (%d × %d) regular grid.'], ...
                        size(T, 1), Texpected, opts.nx, opts.ny);
            end

        case 'manual'          % pass-through
            if ~isfield(opts, 'V') || ~isfield(opts, 'T')
                error('simplex:triangulate:missingManual', ...
                      '''manual'' mode requires opts.V and opts.T.');
            end
            V = opts.V;
            T = opts.T;
            if size(V, 2) ~= 2
                error('simplex:triangulate:badManualV', ...
                      'opts.V must have 2 columns (got %d).', size(V, 2));
            end
            if size(T, 2) ~= 3
                error('simplex:triangulate:badManualT', ...
                      'opts.T must have 3 columns (got %d).', size(T, 2));
            end

        otherwise
            error('simplex:triangulate:badMode', ...
                  'Unknown mode ''%s''. Use ''bounding'', ''delaunay'', or ''manual''.', mode);
    end
end
