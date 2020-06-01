function tileInputAndMaps(img, maps, tickLabels, cmap, tileTitle)
    % Copyright 2019 - 2020 The MathWorks, Inc.

    % This function creates a tiled image using `tiledlayout` to create a row of images.
    % The supplied image is the first element and the rest is the supplied saliency maps,
    % visualised using the color map `cmap`.
    
    arguments
        img % Input image used for the saliency maps `maps`
        maps % A cell array of saliency maps
        tickLabels % The title for each saliency map
        cmap % A color map used to show the saliency maps
        tileTitle = "" % An overall title for the visualisation
    end
    t = tiledlayout(1, length(maps)+1, 'Padding', 'none');
    % title(t, tileTitle)
    xlabel(t, tileTitle);

    nexttile
    imshow(img)
    title(tickLabels{1})
    
    for i=1:length(maps)
        nexttile
        imshow(maps{i}, cmap)
        title(tickLabels{i+1})
    end

end