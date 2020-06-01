function [rand_layer, wasRandomized] = randomizeLayer(layer)
    % Copyright 2019 - 2020 The MathWorks, Inc.
    %
    % Check the layer type and determine if we can randomise it
    
    rand_layer = layer;
    if isprop(layer,'Weights') && isprop(layer,'Bias')
        rand_layer.Weights = 0.01 * randn(size(layer.Weights));
        rand_layer.Bias = 0.01 * randn(size(layer.Bias));
        wasRandomized = true;
    else
        wasRandomized = false;
    end

end

