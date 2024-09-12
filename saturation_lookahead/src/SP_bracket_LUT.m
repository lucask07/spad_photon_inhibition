classdef SP_bracket_LUT
%   A simple hashtable-based look-up table to convert bracketed measurements from the single-photon
%   sensor to the corresponding MLE of incident flux, obtained numerically. 
%   See src/SP_bracket_MLE.m for more details on that.
    properties
        ExposuresRep
        Exposures
        MaxExposure
        PossibleObs
        PossibleObsMLE
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
            obs = reshape(obs, [prod(sz_orig) numel(obj.Exposures)]);
            key = arrayfun(@(i) keyHash(obs(i,:)), (1:size(obs,1))', ...
                            UniformOutput=true);
            key = reshape(key, sz_orig);
        end
    end

    methods
        function obj = SP_bracket_LUT(seq)
            arguments
                seq (1,:)
            end

            obj.ExposuresRep = seq;
            obj.Exposures = unique(seq);
            obj.MaxExposure = sum(seq);
            
            obj.PossibleObs = SP_bracket_possObs(obj.ExposuresRep);
            obj.PossibleObsMLE = SP_bracket_MLE(obj.ExposuresRep, obj.PossibleObs);

            obj = obj.fill_map();
        end

        function phi_mle = mle_lookup(obj, obs)
            phi_mle = obj.MLE_Map(obj.encodeObs(obs));
        end
    end
end
