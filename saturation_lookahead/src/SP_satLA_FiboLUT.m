classdef SP_satLA_FiboLUT < SP_satLA_LUT
    % A specialized look-up table for Fibonacci brackets + saturation look-ahead inhibition,
    % with the saturation thresholds set as [2,1,1,1,...]
    % Instead of using a generic hash function, a simple encoding function is derived based on the
    % special structure of the set of possible measurements in this setup, which can be implemented
    % much more efficiently (ref. the encodeObs method).
    % See src/SatLA_LUT_encoding.pdf for a description.
    properties
        MapSize
    end
    
    methods (Access = protected)
        function obj = fill_map(obj)
            d = ndims(obj.PossibleObs);
            assert(size(obj.PossibleObs, d) == 2*numel(obj.Exposures));

            all_keys = obj.encodeObs(obj.PossibleObs);
            obj.MLE_Map = -ones([obj.MapSize 1], "double");
            obj.MLE_Map(all_keys) = obj.PossibleObsMLE(:);
        end

        function key = encodeObs(obj, obs)
            nd = ndims(obs);
            sz_orig = size(obs, 1:nd-1);
            if isscalar(sz_orig)
                sz_orig = [sz_orig 1];
            end
            obs = reshape(obs, [prod(sz_orig) 2*numel(obj.Exposures)]);
            D = obs(:,1:2:end); % #dets. with each exposure time
            M = obs(:,2:2:end); % whether the corresponding exposure time was enabled

            % Encoding the total #measurements is enough to recover the inhibition pattern here
            % 2 measurements => all exposures {2,3,5,8,...} were inhibited.
            % 3 => {3,5,8,..} were inhibited, and so on.
            numM = uint8(sum(M, 2));
            % For the longer exposures {2,3,5,8,...} all non-inhibited measurements have to be 1, 
            % so we don't need to explicitly encode them.
            % The only possible variation is in the first two exposures {1,1}, which could be 
            % 0, 1, or 2. But with the threshold set at 2 the last case will reflect in
            % #measurements already, so doesn't need to be encoded separately.
            % So we just need to know whether we saw an odd or an even #detections in the first
            % exposure
            D12 = mod(D(:,1), 2);
            % The other possible uncertainty is if we didn't see saturation anywhere and got to the
            % last measurement (the longest exposure), which could be 0 or 1.
            last = true([numel(D12) 1]);
            ind_maxexp = (numM == numel(obj.ExposuresRep));
            last(ind_maxexp) = (D(ind_maxexp, end) > 0);

            % We squeeze the above three things into a single uint8, assuming the bracket is not
            % too long (for the one in the paper this is 5 bits in total)
            key = reshape(uint8(1 + ((numM-1)*4 + D12*2 + uint8(last))), ...
                        sz_orig);
        end
    end

    methods
        function obj = SP_satLA_FiboLUT(seq, thr, opts)
            %SP_SATLA_FIBOLUT
            arguments
                seq (1,:)
                thr (1,:)
                opts.map_size (1,1) {mustBePositive,mustBeInteger} = 32
            end

            obj = obj@SP_satLA_LUT(seq, thr);
            obj.MapSize = opts.map_size;
        end
    end
end

