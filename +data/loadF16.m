function D = loadF16(dataPath)
% LOADF16 Load and unpack the F-16 measurement dataset.
% Inputs: dataPath is the assignment folder; if empty, use the current project.
% Outputs: D contains Cm, measured states, inputs, dt, N and time vector.

    if nargin < 1 || isempty(dataPath)
        thisFile = mfilename('fullpath');
        thisDir  = fileparts(thisFile);              % .../code/+data
        codeDir  = fileparts(thisDir);                % .../code
        dataPath = fileparts(codeDir);                % .../assignment1_multivariate_splines
    end

    matFile = fullfile(dataPath, 'F16traindata_CMabV_2026.mat');
    if ~isfile(matFile)
        error('data:loadF16:fileNotFound', ...
              'Could not find %s. Pass the correct dataPath.', matFile);
    end

    S = load(matFile, 'Cm', 'Z_k', 'U_k');

    % Defensive size checks because the .mat is supposed to be as documented
    assert(size(S.Z_k,2) == 3, 'Z_k should be N×3 (α_m, β_m, V_m).');
    assert(size(S.U_k,2) == 3, 'U_k should be N×3 (u̇, v̇, ẇ).');
    assert(size(S.Cm,1)  == size(S.Z_k,1), 'Cm and Z_k length mismatch.');
    assert(size(S.Cm,1)  == size(S.U_k,1), 'Cm and U_k length mismatch.');

    N  = size(S.Cm, 1);
    dt = 0.01;                          % PDF Appendix B (KF integration step)

    D.Cm      = S.Cm(:);
    D.alpha_m = S.Z_k(:,1);
    D.beta_m  = S.Z_k(:,2);
    D.V_m     = S.Z_k(:,3);
    D.U_k     = S.U_k;
    D.dt      = dt;
    D.N       = N;
    D.t       = (0:N-1).' * dt;
end
