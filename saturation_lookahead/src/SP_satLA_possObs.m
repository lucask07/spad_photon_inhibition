function [obs] = SP_satLA_possObs(seq, thr)
%SP_SATLA_POSSOBS   Generating all possible observations from a saturation lookahead policy
%                   Input "seq" is assumed to be of the form
%                           [T1, T1, T1,    T2 T2 T2,   T3, T3],
%                   if there are 3 distinct exposure times T1, T2, T3, and 3 measurements of T1 and 
%                   T2 are taken, and 2 for T3.
%                   "thr" would be of the form [t1, t2], thresholding #detections for T1 and T2 
%                   in this case.
%
%                   Result is of the form [D1, M1, D2, M2, ...] where 
%                           D1 = #detections with T1, and
%                           M1 = #measurements with T1
%
%   See SP_bracket_possObs.m for a description of the overall recursive approach.
arguments
    seq (1,:)   {mustBePositive}
    thr (1,:)   {mustBeInteger,mustBeNonnegative}
end
    
    seq_u = sort(unique(seq));
    M = numel(seq_u);
    N_u = arrayfun(@(s) sum(seq == s), seq_u, UniformOutput=true);
    obs = zeros([0 2*M], "uint8");
    assert(numel(thr) == M - 1);

    n = N_u(1);
    if M == 1
        % the single-exposure-time setting behaves exactly the same as in SP_bracket_possObs
        obs = [uint8([(0:n)', n*ones([n+1 1])])];
    else
        t = thr(1);
        % We divide the set into two possibilities.
        % The first set doesn't have saturation => the remaining brackets are enabled, 
        % hence a similar recursive call as SP_bracket_possObs.
        obs_rem_enabled = SP_satLA_possObs(seq(n+1:end), thr(2:end));
        for k = 0:min(n,t-1)
            for j = 1:size(obs_rem_enabled,1)
                obs = [obs; ...
                       [[k, n], obs_rem_enabled(j,:)]]; %#ok<AGROW>
            end
        end
        % For the other case the future observations are inhibited, so we can directly fill in zeros
        % throughout and return (the enable signal will also be set to 0 to signify the absence of
        % measurements to downstream methods).
        obs_rem_disabled = zeros([1 2*(M-1)], "uint8");
        for k = t:n
            obs = [obs; ...
                   [[k, n], obs_rem_disabled]]; %#ok<AGROW>
        end
    end
end
