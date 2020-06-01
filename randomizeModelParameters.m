function [model, layersRandomized] = randomizeModelParameters(model, start, count)
    % Copyright 2019 - 2020 The MathWorks, Inc.
    % randomizeModelParameters  function to randomise the weights of 'model'
    
    % This function randomizes the parameters of a model. It accepts a 'SeriesNetwork' or 'DAGNetwork' as input.
    % It assumes that start refers to the layer starting from the beginning of the network
    % and count is the number of extra layers you want to randomise

    % Example:
    %     randModel = randomizeModelParameters(model, 1, 4) % randomise the first 5 layers of the model
    arguments
        model
        start (1,1) {mustBeNumeric} = 1
        count (1,1) {mustBeNumeric} = 0
    end

    layers = model.Layers;
    if any(strcmp(class(model), 'SeriesNetwork'))
        lGraph = layerGraph(model.Layers);
    else
        lGraph = layerGraph(model);
    end
    
    layersRandomized = {};
    for i=start:count + start
        [layers(i), wasRandomized] = randomizeLayer(layers(i));
        if wasRandomized
            lGraph = replaceLayer(lGraph, lGraph.Layers(i).Name, layers(i));
            layersRandomized{end+1} = layers(i).Name;
        end
    end

    model = assembleNetwork(lGraph);

end 
