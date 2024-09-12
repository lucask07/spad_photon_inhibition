classdef SP_satLA_LUT
    % A hashtable-based LUT very similar to SP_bracket_LUT, except we enumerate possibilities using
    % SP_satLA_possObs.m instead of SP_bracket_possObs.m, and same for MLE.
    %
    % A more specialized implementation for Fibonacci brackets is given in SP_satLA_FiboLUT.m, which
    % is more efficient than this for that setting.
    properties
        ExposuresRep
        Exposures
        MaxExposure
        Thresholds
        PossibleObs
        PossibleObsMLE
        ObsKeys
        MLE_Map
    end

    methods (Access = protected)
        function obj = fill_map(obj)
            d = ndims(obj.PossibleObs);
            assert(size(obj.PossibleObs, d) == numel(obj.Exposures));

            all_keys = obj.encodeObs(obj.PossibleObs);
            obj.MLE_Map = dictionary(all_keys(:), obj.PossibleObsMLE(:));
        end

        function key = encodeObs(obj, obs)
            nd = ndims(obs);
            sz_orig = size(obs, 1:nd-1);
            if isscalar(sz_orig)
                sz_orig = [sz_orig 1];
            end
            obs = reshape(obs, [prod(sz_orig) 2*numel(obj.Exposures)]);
            key = arrayfun(@(i) keyHash(obs(i,:)), (1:size(obs,1))', ...
                            UniformOutput=true);
            key = reshape(key, sz_orig);
        end
    end

    methods
        function obj = SP_satLA_LUT(seq, thr)
            %SP_SATLA_FIBOLUT
            arguments
                seq (1,:)
                thr (1,:)
            end

            obj.ExposuresRep = seq;
            obj.Exposures = unique(seq);
            obj.Thresholds = thr;
            obj.MaxExposure = sum(seq);
            
            obj.PossibleObs = SP_satLA_possObs(obj.ExposuresRep, obj.Thresholds);
            obj.PossibleObsMLE = SP_satLA_MLE(obj.ExposuresRep, obj.PossibleObs);

            obj = obj.fill_map();
        end

        function phi_mle = mle_lookup(obj, obs)
            phi_mle = obj.MLE_Map(obj.encodeObs(obs));
        end
    end
end
