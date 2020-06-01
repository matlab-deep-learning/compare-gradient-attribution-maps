classdef AttributionReluLayer < nnet.layer.Layer
    % AttributionReluLayer  Relu layer with customizable backward

    % Copyright 2019 - 2020 The MathWorks, Inc.
    %
    % This custom layer is intended for use in gradient attribution
    % visualizations. It implements standard backprop, Zeiler-Fergus
    % backprop and guided backprop.
    %
    % Example:
    %   layer = AttributionReluLayer;
    %   layer.BackpropMode = "zeiler-fergus";    
    
    %   Copyright 2019 - 2020 The MathWorks, Inc.

    properties
        BackpropMode (1,1) string {mustBeMember(BackpropMode,{'Backprop','GuidedBackprop','ZeilerFergus'})} = 'GuidedBackprop'
    end
    
    methods        
        function Z = predict(~, X)
            % Forward pass is usual ReLu function
            
            Z = max(X, 0);
        end

        function dLdX = backward(layer, X, ~, dLdZ, ~)
            % Backward pass can be modified from the conventional ReLU
            % backward.
            
            switch layer.BackpropMode
                case "ZeilerFergus"
                    dLdX = (dLdZ > 0) .* dLdZ;
                case "GuidedBackprop"
                    dLdX = (X > 0) .* (dLdZ > 0) .* dLdZ;
                case "Backprop"
                    dLdX = dLdZ;
            end

        end
    end
end