import ee
import geemap
import datetime
import os
import requests
from loguru import logger

def initialize_earth_engine():
    """
    Initialize Earth Engine with your project ID.
    Make sure you have authenticated with `earthengine authenticate`.
    """
    ee.Initialize(project='brainbox-448715')  # Replace with your own project ID if needed

def maskS2clouds(image):
    """
    Mask clouds using the Sentinel-2 QA band.
    Bits 10 and 11 are clouds and cirrus, respectively.
    """
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # 0 = clear
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    # Update mask and scale reflectance to 0-1
    return image.updateMask(mask).divide(10000)

def convert_geojson_to_ee(geojson_path):
    """
    Convert a GeoJSON file (local path) to an Earth Engine FeatureCollection,
    removing the system:index property to avoid type errors.
    """
    fc = geemap.geojson_to_ee(geojson_path)
    def fix_index(feature):
        # Remove system:index property altogether or set to empty string
        return feature.set('system:index', '')
    return fc.map(fix_index)

def retrieve_sentinel2_mosaic(roi, start_date, end_date):
    """
    Retrieve a median mosaic of cloud-masked Sentinel-2 images within date range.
    This approach merges multiple images to reduce cloud coverage.
    """
    collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterBounds(roi)
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        .map(maskS2clouds))
    
    # Median composite across all images in the collection
    mosaic = collection.median()
    return mosaic

def calculate_indices(sentinel2_mosaic):
    """
    Calculate NDVI, NDMI, and NDWI from the mosaic image.
    """
    ndvi = sentinel2_mosaic.normalizedDifference(['B8', 'B4'])
    ndmi = sentinel2_mosaic.normalizedDifference(['B8', 'B11'])
    ndwi = sentinel2_mosaic.normalizedDifference(['B3', 'B8'])
    return ndvi, ndmi, ndwi

def generate_unique_filename(file_prefix, folder, extension):
    """
    Generate a unique filename based on current date/time.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_name = f'{file_prefix}_{timestamp}.{extension}'
    return os.path.join(folder, unique_name)

def export_image_to_geotiff(image, filename, scale, region):
    """
    Export an image to a GeoTIFF file using geemap.ee_export_image().
    """
    geemap.ee_export_image(
        image,
        filename=filename,
        scale=scale,
        region=region.geometry()
    )

def export_image_to_png(image, filename, dimensions, region, viz_params=None):
    """
    Export an image to a PNG using Earth Engine's thumbnail API.
    'dimensions' sets the max pixel width/height (e.g., "2048").
    'viz_params' can include 'min', 'max', 'palette', etc.
    """
    bbox = region.geometry().bounds().getInfo()['coordinates']
    params = {
        'region': bbox,
        'dimensions': dimensions,
        'format': 'png'
    }
    if viz_params:
        params.update(viz_params)
    url = image.getThumbURL(params)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def export_imagery(geojson_path, start_date, end_date, output_folder):
    """
    Main function to export NDVI, NDMI, NDWI, and RGB images as both TIFF and PNG.
    Uses a median mosaic of cloud-masked Sentinel-2 data for clearer results.
    - scale=5 for TIFF for higher resolution on a small ROI
    - PNG dimension=2048 to avoid hitting the 50 MB request limit
    """
    initialize_earth_engine()
    roi = convert_geojson_to_ee(geojson_path)

    # Build the median mosaic of cloud-masked images
    mosaic = retrieve_sentinel2_mosaic(roi, start_date, end_date)
    if not mosaic:
        logger.warning("No Sentinel-2 mosaic found in the given date range.")
        return {}

    # Compute NDVI, NDMI, NDWI
    ndvi, ndmi, ndwi = calculate_indices(mosaic)
    
    # Clip them to the ROI
    ndvi = ndvi.clip(roi)
    ndmi = ndmi.clip(roi)
    ndwi = ndwi.clip(roi)
    rgb_image = mosaic.select(['B4','B3','B2']).clip(roi)

    # Generate TIFF filenames
    ndvi_tif = generate_unique_filename('Sentinel2_NDVI_Image', output_folder, 'tif')
    ndmi_tif = generate_unique_filename('Sentinel2_NDMI_Image', output_folder, 'tif')
    ndwi_tif = generate_unique_filename('Sentinel2_NDWI_Image', output_folder, 'tif')
    rgb_tif  = generate_unique_filename('Sentinel2_RGB_Image',  output_folder, 'tif')

    # Export TIFF at scale=5
    scale = 5
    export_image_to_geotiff(ndvi, ndvi_tif, scale=scale, region=roi)
    export_image_to_geotiff(ndmi, ndmi_tif, scale=scale, region=roi)
    export_image_to_geotiff(ndwi, ndwi_tif, scale=scale, region=roi)
    export_image_to_geotiff(rgb_image, rgb_tif, scale=scale, region=roi)

    # Generate PNG filenames
    ndvi_png = generate_unique_filename('Sentinel2_NDVI_Image', output_folder, 'png')
    ndmi_png = generate_unique_filename('Sentinel2_NDMI_Image', output_folder, 'png')
    ndwi_png = generate_unique_filename('Sentinel2_NDWI_Image', output_folder, 'png')
    rgb_png  = generate_unique_filename('Sentinel2_RGB_Image',  output_folder, 'png')

    # Visualization parameters
    ndvi_viz = {
        'min': -0.2, 'max': 0.8,
        'palette': ['#8b0000', '#ff0000', '#ffff00', '#00ff00', '#006400']
    }
    ndmi_viz = {
        'min': -0.2, 'max': 0.8,
        'palette': ['#7f3b08','#fdd49e','#ffffbf','#abd9e9','#2c7bb6']
    }
    ndwi_viz = {
        'min': -1, 'max': 1,
        'palette': ['#ffffb2','#fed976','#feb24c','#fd8d3c','#f03b20','#bd0026']
    }

    # Use "2048" to keep request size under 50 MB
    dim = "2048"
    export_image_to_png(ndvi, ndvi_png, dimensions=dim, region=roi, viz_params=ndvi_viz)
    export_image_to_png(ndmi, ndmi_png, dimensions=dim, region=roi, viz_params=ndmi_viz)
    export_image_to_png(ndwi, ndwi_png, dimensions=dim, region=roi, viz_params=ndwi_viz)

    # For the RGB mosaic, we visualize with a typical reflectance stretch
    rgb_vis = rgb_image.visualize(min=0, max=0.3)
    export_image_to_png(rgb_vis, rgb_png, dimensions=dim, region=roi)

    logger.info(f"Median mosaic used with scale={scale} for TIFF, dimensions={dim} for PNG.")
    logger.info(f"NDVI TIFF: {ndvi_tif} | PNG: {ndvi_png}")
    logger.info(f"NDMI TIFF: {ndmi_tif} | PNG: {ndmi_png}")
    logger.info(f"NDWI TIFF: {ndwi_tif} | PNG: {ndwi_png}")
    logger.info(f"RGB  TIFF: {rgb_tif}  | PNG: {rgb_png}")

    return {
        'ndvi_tif': ndvi_tif,
        'ndmi_tif': ndmi_tif,
        'ndwi_tif': ndwi_tif,
        'rgb_tif':  rgb_tif,
        'ndvi_png': ndvi_png,
        'ndmi_png': ndmi_png,
        'ndwi_png': ndwi_png,
        'rgb_png':  rgb_png,
    }

