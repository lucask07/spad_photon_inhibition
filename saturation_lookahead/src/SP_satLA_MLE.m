function [phi_mle] = SP_satLA_MLE(seq, obs, opts)
%SP_SATLA_MLE A brute-force numerical estimation of MLE for all possible observation patterns
arguments
    seq (1,:)   {mustBePositive}
    obs         {mustBeInteger}
    opts.num_test (1,1) {mustBePositive,mustBeInteger} = 2000
end
    seq_u = unique(seq);
    assert(size(obs, ndims(obs)) == 2*numel(seq_u));
    sz_obs_orig = [size(obs, 1:ndims(obs)-1) 1];
    obs = reshape(obs, [], 2*numel(seq_u));

    phi_test = [0 linspace(0.001, 10, opts.num_test-1)]';
    phi_seq = phi_test .* seq_u;
    prob1_seq = SP_PPD(phi_seq);
    phi_mle = zeros([size(obs,1) 1], "double");
    for i = 1:size(obs,1)
        D = obs(i,1:2:end);
        N = obs(i,2:2:end);
        prob_D = prod(binopdf(D(ones(opts.num_test,1),:), ...
                                N(ones(opts.num_test,1),:), ...
                                prob1_seq),     2);
        [~, jmax] = max(prob_D);
        phi_mle(i) = phi_test(jmax);
    end

    phi_mle = reshape(phi_mle, sz_obs_orig);
end
