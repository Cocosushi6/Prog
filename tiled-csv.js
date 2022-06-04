
var CSVformat = {
	name: "CSV map format",
    extension: "csv",

    write: function(map, fileName) {
		var finalString = "";

        for (var i = 0; i < map.layerCount; ++i) {
            var layer = map.layerAt(i);
            if (layer.isTileLayer) {
                for (var y = 0; y < layer.height; ++y) {
                    for (var x = 0; x < layer.width; ++x) {
						var id = layer.cellAt(x, y).tileId;
						finalString += id;
						if(x != layer.width - 1) {
							finalString += ",";
						}
					}
                    finalString += "\n"
                }
            }
        }

        var file = new TextFile(fileName, TextFile.WriteOnly);
        file.write(finalString);
        file.commit();
    },

	read: function(fileName) {
		var file = new TextFile(fileName);
		var content = file.readAll();
		file.close();

		let lines = content.split("\n");
		let size = lines.length;

		var map = new TileMap();
		map.setSize(size, size);
		map.setTileSize(32, 32);
		map.orientation = TileMap.Orthogonal;

		var tileset = new Tileset('CSV tileset');
		tileset.setTileSize(32, 32);
		var img = new Image();
		img.load("tileset.png", "png");
		tileset.loadFromImage(img);
		map.addTileset(tileset);

		var tileLayer = new TileLayer();
		var tle = tileLayer.edit();
		for(var x = 0; x < size; x++) {
			var line = lines[x];
			for(var y = 0; y < size; y++) {
				var id = parseInt(line[y]);
				tle.setTile(x, y, tileset.tile(id));
			}
		}
		tle.apply();
		map.addLayer(tileLayer);

		return map;
	}
};

tiled.registerMapFormat("csv", CSVformat);
