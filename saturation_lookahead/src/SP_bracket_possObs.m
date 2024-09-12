function [obs] = SP_bracket_possObs(seq)
%   Enumerating all possible observations for a given exposure bracketing sequence.
%   This is quite feasible for the relatively short sequences we have used in this paper.
arguments
    seq (1,:)   {mustBePositive}
end
    %   Consider the running example [1 1 1 2 2 2 10 10]
    seq_u = sort(unique(seq));          % [1 2 10]
    M = numel(seq_u);                   % 3
    N_u = arrayfun(@(s) sum(seq == s), seq_u, UniformOutput=true);  %   [3 3 2]
    obs = zeros([0 M], "uint8");        % this will contain the whole set of possibilities, and will
                                        % be filled in recursively below
    n = N_u(1);
    if M == 1
        obs = uint8((0:n)');            % all possible #detections for a single exposure time
    else
        % recursively enumerate observations from the tail (cdr) of the sequence, 
        % appending the thing above to those
        obs_rem = SP_bracket_possObs(seq(n+1:end));
        for k = 0:n
            for j = 1:size(obs_rem,1)
                obs = [obs; ...
                       [k, obs_rem(j,:)]]; %#ok<AGROW>
            end
        end
    end
    % For the example this should result in 4x4x3 = 48 combinations as follows:
    %   [(0 to 3) (0 to 3) (0 to 2)], each of length 3, resulting in a 48x3 array
end
