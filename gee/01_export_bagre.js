/**
 * Bagre Sentinel-2 / ESA WorldCover export for the segmentation MVP.
 *
 * Run in https://code.earthengine.google.com/
 * 1. Inspect the AOI and RGB preview before exporting.
 * 2. Adjust AOI only if the irrigation scheme is not well covered.
 * 3. Run both tasks from the Tasks tab.
 * 4. Move the two GeoTIFFs into bagre-segmentation/raw in Google Drive.
 *
 * Approximate Bagre village centre: 11.490 N, 0.523 W.
 * The rectangle is a starting AOI, not an official irrigation-scheme boundary.
 */

var AOI = ee.Geometry.Rectangle([-0.75, 11.25, -0.35, 11.65]);
var START = '2021-01-01';
var END = '2021-05-01'; // Exclusive: January-April dry-season composite.
var BANDS = ['B2', 'B3', 'B4', 'B8'];
var CRS = 'EPSG:32630'; // WGS 84 / UTM Zone 30N.
var SCALE = 10;
var CLOUD_SCORE_THRESHOLD = 0.60;

Map.centerObject(AOI, 9);
Map.addLayer(AOI, {color: 'yellow'}, 'Starting AOI', false);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80));

var cloudScore = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var withCloudScore = s2.linkCollection(cloudScore, ['cs']);

var clearImages = withCloudScore.map(function(image) {
  var scl = image.select('SCL');
  var sclClear = scl.neq(3)   // Cloud shadow.
    .and(scl.neq(8))          // Cloud, medium probability.
    .and(scl.neq(9))          // Cloud, high probability.
    .and(scl.neq(10))         // Thin cirrus.
    .and(scl.neq(11));        // Snow or ice.
  var clear = image.select('cs').gte(CLOUD_SCORE_THRESHOLD).and(sclClear);
  return image.updateMask(clear);
});

print('Sentinel-2 observations before cloud masking:', s2.size());

var composite = clearImages.select(BANDS).median()
  .multiply(0.0001)
  .rename(BANDS)
  .toFloat()
  .clip(AOI);

var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map')
  .clip(AOI);

Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30}, 'Sentinel-2 RGB');
Map.addLayer(worldcover, {}, 'ESA WorldCover 2021', false);

Export.image.toDrive({
  image: composite,
  description: 'bagre_s2_2021_dryseason',
  folder: 'bagre-segmentation',
  fileNamePrefix: 'bagre_s2_2021_dryseason',
  region: AOI,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: -9999}
});

Export.image.toDrive({
  image: worldcover,
  description: 'bagre_worldcover_2021',
  folder: 'bagre-segmentation',
  fileNamePrefix: 'bagre_worldcover_2021',
  region: AOI,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: 0}
});
