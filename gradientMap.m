function maps = gradientMap(model, image, attributionTypes)
    % Copyright 2019 - 2020 The MathWorks, Inc.
    % 
    % Computes a saliency map for `image` using the `model` and the techniques specified
    % in `atrributionTypes`
    %  
    % The number of saliency maps generated is equal to the number of techniques specified in `attributionTypes`
    % as a map is generated for each one.
    %
    % Supported types for `model` are SeriesNetwork, DAGNetworks and dlnetwork
    % 
    if isa(attributionTypes, 'char')
        maps = iGradientMap(model, image, attributionTypes);
    elseif isa(attributionTypes, 'cell')
        maps = arrayfun(@(x) iGradientMap(model, image, x), attributionTypes,'UniformOutput',false);
    end
end



function map = iGradientMap(model, image, attributionType)
    % Computes a saliency map for the image using the model. 
    % `attributionType` determines the technique used to compute the saliency map
    arguments
        model
        image
        attributionType string {mustBeMember(attributionType,{'GradientExplanation','GuidedBackprop','ZeilerFergus'})} = 'GuidedBackprop'
    end

    % DAGNetwork are the supported network type. Therefore, SeriesNetwork needs to be converted
    assert(...
        isa(model,'SeriesNetwork') || isa(model,'DAGNetwork') || isa(model,'dlnetwork'), ...
        "Model must be a SeriesNetwork or DAGNetwork" );
    if (isa(model, 'SeriesNetwork'))
        lgraph = layerGraph(model.Layers);
    else %assume it is a DAGNetwork
        lgraph = layerGraph(model);
    end
    
    lastLayer = lgraph.Layers(end);
    attributionLayer = AttributionReluLayer;

    % Remove the classification layer if any, to enable automatic differentiation
    if any(strcmp(class(lastLayer), {'nnet.cnn.layer.ClassificationOutputLayer','nnet.cnn.layer.RegressionOutputLayer'}))
        lgraph = removeLayers(lgraph, lastLayer.Name);
    end
    
    if canUseGPU
        gpuImg = gpuArray(single(image));
        dlImg = dlarray(gpuImg, 'SSC');
    else            
        dlImg = dlarray(single(image),'SSC');
    end
    
    if strcmp(attributionType,'GradientExplanation')
        %Continue
    elseif any(strcmp({'GuidedBackprop','ZeilerFergus'},attributionType))
        attributionLayer.BackpropMode = attributionType;
        lgraph = replaceLayersWith(lgraph, 'nnet.cnn.layer.ReLULayer', attributionLayer);                
    end

    dlnet = dlnetwork(lgraph);
    dYdI = dlfeval(@gradientExplanation, dlnet, dlImg);
    map = uint8(255 * rescale(sum(abs(extractdata(dYdI)),3)));

    map = gather(map);
end

function lgraph = replaceLayersWith(lgraph, layerType, nlayer)
    for i=1:length(lgraph.Layers)
        if isa(lgraph.Layers(i), layerType)
            %copy over the name first before replacing
            currentName = lgraph.Layers(i).Name;
            nlayer.Name = currentName;

            lgraph = replaceLayer(lgraph, currentName, nlayer);
        end
    end
end

function dYdI = gradientExplanation(dlnet, dlImg)
    Y = predict(dlnet, dlImg);
    dYdI = dlgradient(max(Y), dlImg);
end 
