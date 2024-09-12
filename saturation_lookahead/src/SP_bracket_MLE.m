function [phi_mle] = SP_bracket_MLE(seq, obs, opts)
%SP_BRACKET_MLE A brute-force numerical estimation of MLE for all possible observation patterns
arguments
    seq (1,:)   {mustBePositive}
    obs         {mustBeInteger}
    opts.num_test (1,1) {mustBePositive,mustBeInteger} = 2000
end

    seq_u = unique(seq);
    N_u = arrayfun(@(s) sum(seq == s), seq_u, UniformOutput=true);
    assert(size(obs, ndims(obs)) == numel(seq_u));
    sz_obs = size(obs, 1:ndims(obs)-1);
    if isscalar(sz_obs)
        sz_obs = [sz_obs 1];
    end
    
    obs = reshape(obs, [prod(sz_obs) numel(seq_u)]);

    phi_test = [0 linspace(0.001, 10, opts.num_test-1)]';
    phi_seq = phi_test .* seq_u;
    prob1_seq = SP_PPD(phi_seq);
    phi_mle = zeros([size(obs,1) 1], "double");
    for i = 1:size(obs,1)
        D = obs(i,:);
        prob_D = prod(binopdf(D(ones(opts.num_test,1),:), ...
                                N_u(ones(opts.num_test,1),:), ...
                                prob1_seq),     2);
        [~, jmax] = max(prob_D);
        phi_mle(i) = phi_test(jmax);
    end
    phi_mle = reshape(phi_mle, sz_obs);
end
