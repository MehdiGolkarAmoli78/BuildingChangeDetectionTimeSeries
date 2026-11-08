"""
Sentinel-2 Time Series Building Classification
VERSION 06 - GEE Cloud Masking + Gap-Filling + Python Patch Validation + Zero-Value Preprocessing

ALGORITHM (matching JS code):
═══════════════════════════════════════════════════════════════════════════════
GEE SERVER-SIDE:
1. Filter images by CLOUDY_PIXEL_PERCENTAGE metadata (<10%)
2. Join S2_SR with S2_CLOUD_PROBABILITY
3. Apply cloud mask: cloudProb > 65 AND cdi < -0.5, dilate 20m
4. For each month M:
   a. Create median composite from cloud-free collection
   b. Count masked pixels (frequency == 0)
   c. Status logic:
      - image_count == 0        → "no_data"   → SKIP
      - masked_percent > 30%    → "skipped"   → SKIP (don't try gap-fill)
      - masked_percent == 0%    → "complete"  → DOWNLOAD ✅
      - 0% < masked <= 30%      → TRY GAP-FILL from M-1, M+1:
          * Collect cloud-free images from M-1 and M+1
          * Sort by time distance to 15th of month M
          * Fill gaps with closest cloud-free pixel (mosaic)
          * If masked_after == 0% → "complete" → DOWNLOAD ✅
          * If masked_after > 0%  → "rejected" → SKIP

PYTHON CLIENT-SIDE (after download):
5. **NEW**: Preprocess to check zero-value pixels
   - If >2% of image has zeros → REJECT
   - Otherwise add 1e-4 to zero values in specific bands
6. Validate all downloaded images have SAME dimensions
7. Check patch validity (NaN/zeros) for each month
8. Find the month with MAXIMUM valid patches (reference)
9. EXCLUDE months that don't have ALL reference patches valid
10. Classify only valid months using reference patch mask

GUARANTEES:
═══════════════════════════════════════════════════════════════════════════════
✅ All downloaded images have 0% masked pixels (GEE-side guarantee)
✅ All downloaded images with <2% zeros have NO zero-value pixels (preprocessing)
✅ All downloaded images have identical dimensions (validated in Python)
✅ All classified months have SAME valid patches (reference mask)
✅ Months with missing patches are EXCLUDED (not reduced to minimum)
✅ Maximum number of valid patches preserved
═══════════════════════════════════════════════════════════════════════════════
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import rasterio
from patchify import patchify, unpatchify
import datetime
import torch
import math
import ee
import tempfile
import requests
import time
import warnings
import sys
import base64
import json
import subprocess
from datetime import date
from PIL import Image
from io import BytesIO
from contextlib import contextmanager
import html

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="پایش تغییرات ساختمان‌ها",
    page_icon="🛰️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium
import segmentation_models_pytorch as smp

# =============================================================================
# CONSTANTS
# =============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

# File validation
MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

# Cloud masking thresholds (from JS)
CLOUD_PROB_THRESHOLD = 50
CDI_THRESHOLD = -0.5
cloudy_pct = 10

# Gap-filling threshold
MAX_MASKED_PERCENT_FOR_GAPFILL = 20

# Zero-value preprocessing thresholds
ZERO_VALUE_REPLACEMENT = 1e-5  # Value to add to zero pixels
MAX_ZERO_PERCENT = 1.0  # Maximum percentage of zeros allowed (2%)

# Status constants
STATUS_NO_DATA = "no_data"
STATUS_SKIPPED = "skipped"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"

# =============================================================================
# Session State Initialization
# =============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'classification_thumbnails' not in st.session_state:
    st.session_state.classification_thumbnails = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processed_months' not in st.session_state:
    st.session_state.processed_months = {}
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = {}
if 'valid_patches_mask' not in st.session_state:
    st.session_state.valid_patches_mask = None
if 'valid_months' not in st.session_state:
    st.session_state.valid_months = {}
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None
# For resume capability
if 'month_analysis_results' not in st.session_state:
    st.session_state.month_analysis_results = {}
if 'failed_downloads' not in st.session_state:
    st.session_state.failed_downloads = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'download_complete' not in st.session_state:
    st.session_state.download_complete = False
if 'cloud_free_collection' not in st.session_state:
    st.session_state.cloud_free_collection = None
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = None
# NEW: For robust processing and region selection
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
# NEW: For change detection
if 'probability_maps' not in st.session_state:
    st.session_state.probability_maps = {}
if 'change_detection_result' not in st.session_state:
    st.session_state.change_detection_result = None
if 'change_timing_map' not in st.session_state:
    st.session_state.change_timing_map = None


# =============================================================================
# Normalization function
# =============================================================================
def normalized(img):
    """Normalize image data to range [0, 1] - global normalization"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val + 1e-5)


# =============================================================================
# File Validation Functions
# =============================================================================
def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        min_size = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes)"
        
        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_band_file(band_file_path, band_name):
    return validate_geotiff_file(band_file_path, expected_bands=1)


# =============================================================================
# Model Download Functions
# =============================================================================
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """Download model from Google Drive"""
    try:
        correct_file_id = "1Fgn4j61XQQEoc_TnaUE5ClWDnzoopDgJ"
        st.info(f"Downloading model...")
        
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id
        ]
        
        for method in methods:
            try:
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    st.success(f"Model downloaded!")
                    return local_filename
            except:
                if os.path.exists(local_filename):
                    os.remove(local_filename)
        
        return None
    except Exception as e:
        return None


# =============================================================================
# Model Loading Function
# =============================================================================
@st.cache_resource
def load_model(model_path):
    """Load the UNet++ model"""
    try:
        device = torch.device('cpu')
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b8',
            encoder_weights=None,  # The checkpoint below supplies all trained weights; skip the ImageNet download.
            in_channels=12,
            classes=1,
            decoder_attention_type='scse'
        ).to(device)

        loaded_object = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            model.load_state_dict(loaded_object['model_state_dict'])
        elif isinstance(loaded_object, dict):
            model.load_state_dict(loaded_object)
        else:
            return None, None

        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# =============================================================================
# Earth Engine Authentication
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except Exception:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)
                
                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)
                
                credentials = ee.ServiceAccountCredentials(key_data['client_email'], key_file.name)
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Authenticated with Service Account"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Authenticated"
        except Exception as auth_error:
            return False, f"Auth failed: {str(auth_error)}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    zone_number = get_utm_zone(longitude)
    return f"EPSG:326{zone_number:02d}" if latitude >= 0 else f"EPSG:327{zone_number:02d}"


# =============================================================================
# GEE CLOUD MASKING + GAP-FILLING (Server-Side)
# =============================================================================

def create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage=10):
    """
    Create cloud-free collection with proper cloud masking.
    Matching JS: indexJoin + maskCloudAndShadow
    """
    # Get S2 SR collection
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(SPECTRAL_BANDS + ['SCL']))
    
    # Get cloud probability collection
    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(extended_start, extended_end))
    
    # Join collections (indexJoin) - matching JS exactly
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )
    
    # Convert joined FeatureCollection to ImageCollection and add cloud band
    def add_cloud_band(feature):
        # Cast feature to image
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img)
    
    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))
    
    # Apply cloud masking (maskCloudAndShadow) - matching JS exactly
    def mask_cloud_and_shadow(img):
        # Get cloud probability band
        cloud_prob = img.select('probability')
        
        # Calculate CDI (Cloud Displacement Index)
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        
        # Cloud mask: prob > 65 AND cdi < -0.5
        is_cloud = cloud_prob.gt(CLOUD_PROB_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        
        # Dilate with 20m kernel, 2 iterations
        kernel = ee.Kernel.circle(radius=20, units='meters')
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        
        # Mask clouds and scale
        masked = img.updateMask(cloud_dilated.Not())
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        
        return scaled.copyProperties(img, ['system:time_start'])
    
    return s2_joined.map(mask_cloud_and_shadow)



# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
    try:
        region = aoi.bounds().getInfo()['coordinates']
    except Exception as e:
        st.error(f"❌ Failed to get AOI bounds: {e}")
        return False, f"AOI bounds error: {e}"
    
    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if os.path.exists(output_path):
        is_valid, msg = validate_band_file(output_path, band)
        if is_valid:
            return True, "cached"
        os.remove(output_path)
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Get download URL
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })
            
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit (HTML response)"
                    raise Exception(last_error)
                
                # Download to temp file
                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                if downloaded_size < MIN_BAND_FILE_SIZE:
                    last_error = f"File too small ({downloaded_size} bytes)"
                    raise Exception(last_error)
                
                is_valid, msg = validate_band_file(temp_path, band)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    last_error = f"Validation failed: {msg}"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(last_error)
            else:
                last_error = f"HTTP {response.status_code}"
                raise Exception(last_error)
                
        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            if last_error is None:
                last_error = str(e)
        
        # Cleanup
        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            time.sleep(wait_time)
    
    return False, last_error


def download_composite(composite, aoi, output_path, month_name, scale=10, status_placeholder=None):
    """Download composite image to GeoTIFF with detailed error reporting."""
    try:
        if os.path.exists(output_path):
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                if status_placeholder:
                    status_placeholder.info(f"✅ {month_name} cached")
                return output_path
            else:
                st.warning(f"⚠️ {month_name}: Cached file invalid ({msg}), re-downloading...")
                os.remove(output_path)
        
        temp_dir = os.path.dirname(output_path)
        bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)
        
        band_files = []
        failed_bands = []
        
        for i, band in enumerate(SPECTRAL_BANDS):
            band_file = os.path.join(bands_dir, f"{band}.tif")
            
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
            
            success, error_msg = download_band_with_retry(composite, band, aoi, band_file, scale)
            
            if success:
                band_files.append(band_file)
            else:
                failed_bands.append(f"{band}: {error_msg}")
        
        if failed_bands:
            st.error(f"❌ {month_name}: Failed bands - {'; '.join(failed_bands)}")
            return None
        
        if len(band_files) == len(SPECTRAL_BANDS):
            if status_placeholder:
                status_placeholder.text(f"📦 {month_name}: Merging bands...")
            
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            meta.update(count=len(band_files))
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            is_valid, msg = validate_geotiff_file(output_path, expected_bands=len(SPECTRAL_BANDS))
            if is_valid:
                return output_path
            else:
                st.error(f"❌ {month_name}: Merged file validation failed: {msg}")
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        return None
        
    except Exception as e:
        st.error(f"❌ {month_name}: Download exception - {str(e)}")
        return None


# =============================================================================
# NEW: Zero-Value Preprocessing Function
# =============================================================================
def preprocess_zero_values(image_path, month_name):
    """
    Check and preprocess zero values in the image.
    
    Algorithm:
    1. Count total zero values across all bands
    2. If >2% of image has zeros → REJECT (return False)
    3. Otherwise, add 1e-4 to pixels with zero values in specific bands
    4. Modify file in-place
    
    Args:
        image_path: Path to the GeoTIFF file
        month_name: Name of the month (for logging)
    
    Returns:
        True if preprocessing successful (or no zeros), False if rejected (>2% zeros)
    """
    try:
        # Read the image
        with rasterio.open(image_path) as src:
            img_data = src.read()  # Shape: (bands, height, width)
            profile = src.profile
        
        # Calculate total number of values
        total_values = img_data.size  # bands * height * width
        
        # Create mask where values are exactly 0
        zero_mask = (img_data == 0)
        total_zeros = np.sum(zero_mask)
        
        # Calculate percentage of zeros
        zero_percent = (total_zeros / total_values) * 100
        
        # CHECK: If >2% zeros, REJECT the image
        if zero_percent > MAX_ZERO_PERCENT:
            st.warning(f"❌ {month_name}: REJECTED - Too many zero values ({zero_percent:.2f}% > {MAX_ZERO_PERCENT}%)")
            return False
        
        # If we have zeros but <=2%, fix them
        if total_zeros > 0:
            # Count zeros per band for logging
            zeros_per_band = np.sum(zero_mask, axis=(1, 2))
            
            # Log which bands have zeros (show first 3 bands)
            bands_with_zeros = []
            for i in range(len(zeros_per_band)):
                if zeros_per_band[i] > 0:
                    bands_with_zeros.append(f"Band{i+1}:{int(zeros_per_band[i])}")
            
            # Add 1e-4 to zero values (only affects pixels with 0 in specific bands)
            img_data[zero_mask] = ZERO_VALUE_REPLACEMENT
            
            # Write corrected data back to file
            with rasterio.open(image_path, 'w', **profile) as dst:
                dst.write(img_data)
            
            # Show abbreviated band list
            band_display = ', '.join(bands_with_zeros[:3])
            if len(bands_with_zeros) > 3:
                band_display += f"... +{len(bands_with_zeros)-3} more"
            
            st.info(f"✅ {month_name}: Fixed {total_zeros} zero pixels ({zero_percent:.2f}%) - {band_display}")
        
        # No zeros or successfully fixed
        return True
            
    except Exception as e:
        st.warning(f"⚠️ {month_name}: Error preprocessing zero values - {str(e)}")
        return False


def download_monthly_image_v06(aoi, cloud_free_collection, month_info, temp_dir, 
                                scale=10, status_placeholder=None):
    """
    Download a single monthly composite with gap-filling and zero-value preprocessing.
    Creates composite fresh inside function (like v05) but with v06 cloud masking.
    
    This function:
    1. Creates monthly median from cloud-free collection
    2. Checks masked pixel percentage
    3. Applies gap-filling if 0% < masked <= 30%
    4. Downloads only if final masked == 0%
    5. **NEW: Preprocesses to check/fix zero values (reject if >2%)**
    
    Returns: (output_path, status, message)
    """
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']
    
    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    
    # Check cache first
    if os.path.exists(output_file):
        is_valid, msg = validate_geotiff_file(output_file, expected_bands=len(SPECTRAL_BANDS))
        if is_valid:
            if status_placeholder:
                status_placeholder.info(f"✅ {month_name} using cached file")
            return output_file, STATUS_COMPLETE, "Cached"
        else:
            if status_placeholder:
                status_placeholder.warning(f"⚠️ {month_name} cache invalid, re-processing...")
            os.remove(output_file)
    
    try:
        origin_date = ee.Date(origin)
        month_start = origin_date.advance(month_index, 'month')
        month_end = origin_date.advance(ee.Number(month_index).add(1), 'month')
        month_middle = month_start.advance(15, 'day')
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Analyzing...")
        
        # Filter to current month
        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size().getInfo()
        
        # CHECK 1: No images
        if image_count == 0:
            return None, STATUS_NO_DATA, "No images available"
        
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Creating composite from {image_count} images...")
        
        # Create frequency map and median composite
        def create_valid_mask(img):
            return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()
        
        frequency = monthly_images.map(create_valid_mask).sum().toInt().rename('frequency')
        composite = monthly_images.median()
        
        # Calculate masked pixel statistics
        masked_stats = frequency.eq(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        )
        total_stats = frequency.gte(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        )
        
        masked_count = ee.Number(masked_stats.get('frequency')).getInfo()
        total_count = ee.Number(total_stats.get('frequency')).getInfo()
        
        if total_count == 0:
            return None, STATUS_NO_DATA, "No valid pixels"
        
        masked_percent = (masked_count / total_count) * 100
        
        # CHECK 2: Too many masked (> 30%)
        if masked_percent > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {masked_percent:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%"
        
        # CHECK 3: No masked pixels - ready to download
        if masked_percent == 0:
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Complete (0% masked), downloading...")
            
            path = download_composite(composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                # NEW: Preprocess to check/fix zero values
                if status_placeholder:
                    status_placeholder.text(f"🔧 {month_name}: Checking zero values...")
                if preprocess_zero_values(path, month_name):
                    return path, STATUS_COMPLETE, "Complete (0% masked)"
                else:
                    # Image rejected due to >2% zeros
                    if os.path.exists(path):
                        os.remove(path)
                    return None, STATUS_REJECTED, "Too many zero values (>2%)"
            else:
                return None, STATUS_REJECTED, "Download failed"
        
        # GAP-FILL: 0% < masked <= 30%
        if status_placeholder:
            status_placeholder.text(f"📥 {month_name}: Gap-filling ({masked_percent:.1f}% masked)...")
        
        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()
        
        # M-1 and M+1 ranges
        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')
        
        # Collect and sort by time distance
        m1_past_images = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
        m1_future_images = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
        all_candidates = m1_past_images.merge(m1_future_images)
        
        def add_time_distance(img):
            img_time = ee.Number(img.get('system:time_start'))
            time_diff = img_time.subtract(month_middle_millis).abs()
            return img.set('time_distance', time_diff)
        
        images_with_distance = all_candidates.map(add_time_distance)
        sorted_images = images_with_distance.sort('time_distance', True)
        
        candidate_count = sorted_images.size().getInfo()
        
        if candidate_count == 0:
            return None, STATUS_REJECTED, f"No gap-fill candidates, {masked_percent:.1f}% still masked"
        
        # Create mosaic (closest pixel first)
        closest_mosaic = sorted_images.mosaic().select(SPECTRAL_BANDS)
        has_closest = closest_mosaic.select('B4').mask()
        
        fill_from_closest = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())
        
        filled_composite = composite.unmask(closest_mosaic.updateMask(fill_from_closest))
        
        # Check remaining masked pixels
        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from_closest, 1)
                       .where(still_masked, 2)
                       .rename('fill_source'))
        
        still_masked_result = fill_source.eq(2).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        ).get('fill_source')
        
        still_masked_count = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(still_masked_result, None), 0, still_masked_result
        )).getInfo()
        
        # CHECK 4: After gap-fill
        if still_masked_count == 0:
            if status_placeholder:
                status_placeholder.text(f"📥 {month_name}: Gap-filled successfully, downloading...")
            
            path = download_composite(filled_composite, aoi, output_file, month_name, scale, status_placeholder)
            if path:
                # NEW: Preprocess to check/fix zero values
                if status_placeholder:
                    status_placeholder.text(f"🔧 {month_name}: Checking zero values...")
                if preprocess_zero_values(path, month_name):
                    return path, STATUS_COMPLETE, f"Complete after gap-fill (was {masked_percent:.1f}%)"
                else:
                    # Image rejected due to >2% zeros
                    if os.path.exists(path):
                        os.remove(path)
                    return None, STATUS_REJECTED, "Too many zero values (>2%) after gap-fill"
            else:
                return None, STATUS_REJECTED, "Download failed after gap-fill"
        else:
            still_masked_pct = (still_masked_count / total_count) * 100
            return None, STATUS_REJECTED, f"{still_masked_pct:.1f}% still masked after gap-fill"
        
    except Exception as e:
        return None, STATUS_NO_DATA, f"Error: {str(e)}"


# =============================================================================
# RGB Thumbnail Generation
# =============================================================================
def generate_rgb_thumbnail(image_path, month_name, max_size=256):
    """Generate RGB thumbnail from Sentinel-2 image."""
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                p_low = np.percentile(valid, lower)
                p_high = np.percentile(valid, upper)
                if p_high <= p_low:
                    p_high = p_low + 0.001
                stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                return (stretched * 255).astype(np.uint8)
            
            rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
            
            pil_img = Image.fromarray(rgb_uint8, mode='RGB')
            
            h, w = pil_img.size[1], pil_img.size[0]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            
            return pil_img
    except:
        return None


# =============================================================================
# PYTHON-SIDE PATCH VALIDITY CHECKING
# =============================================================================
def check_patch_validity(patch, nodata_threshold_percent=0):
    """Check if a patch has minimal nodata."""
    if np.any(np.isnan(patch)):
        return False
    if np.all(patch == 0):
        return False
    
    zero_percent = (np.sum(patch == 0) / patch.size) * 100
    if zero_percent > nodata_threshold_percent:
        return False
    
    if patch.ndim == 3:
        for band_idx in range(patch.shape[-1]):
            if np.all(patch[:, :, band_idx] == 0):
                return False
    
    return True


def get_patch_validity_mask(image_path, patch_size=224, nodata_threshold_percent=0):
    """Create a mask showing which patches are valid for a single image."""
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        new_h = int(np.ceil(h / patch_size) * patch_size)
        new_w = int(np.ceil(w / patch_size) * patch_size)
        
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        patches = patchify(img_for_patching, (patch_size, patch_size, c), step=patch_size)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        validity_mask = np.zeros((n_patches_h, n_patches_w), dtype=bool)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = patches[i, j, 0]
                validity_mask[i, j] = check_patch_validity(patch, nodata_threshold_percent)
        
        return validity_mask, (h, w), (n_patches_h, n_patches_w)
    except:
        return None, None, None


def find_common_valid_patches(downloaded_images, nodata_threshold_percent=0):
    """
    Find patches that are valid and EXCLUDE months that don't have all valid patches.
    
    NEW LOGIC:
    1. Calculate the validity mask for each month
    2. Find the month with the MOST valid patches (reference)
    3. Use that reference mask as the "expected" valid patches
    4. Exclude any month that doesn't have ALL those patches valid
    5. Return the reference mask and list of valid months
    
    Returns: (validity_mask, original_size, valid_months_dict) or (None, None, None) if fails
    """
    st.info("🔍 Analyzing patch validity across all months...")
    
    month_names = sorted(downloaded_images.keys())
    
    if len(month_names) == 0:
        st.error("❌ No downloaded images to analyze!")
        return None, None, None
    
    # =========================================================================
    # STEP 1: Validate all images have the same dimensions
    # =========================================================================
    st.write("**Step 1: Validating image dimensions...**")
    
    dimensions = {}
    for month_name in month_names:
        image_path = downloaded_images[month_name]
        try:
            with rasterio.open(image_path) as src:
                h, w = src.height, src.width
                bands = src.count
                dimensions[month_name] = {'height': h, 'width': w, 'bands': bands}
        except Exception as e:
            st.error(f"❌ Cannot read {month_name}: {e}")
            return None, None, None
    
    # Check if all dimensions match
    first_month = month_names[0]
    reference_dim = dimensions[first_month]
    
    mismatched = []
    for month_name, dim in dimensions.items():
        if dim['height'] != reference_dim['height'] or dim['width'] != reference_dim['width']:
            mismatched.append(f"{month_name}: {dim['height']}x{dim['width']}")
    
    if mismatched:
        st.error(f"❌ **DIMENSION MISMATCH DETECTED!**")
        st.error(f"Reference ({first_month}): {reference_dim['height']}x{reference_dim['width']}")
        st.error(f"Mismatched: {', '.join(mismatched)}")
        st.warning("All months must have identical image dimensions for consistent patch analysis.")
        return None, None, None
    
    st.success(f"✅ All {len(month_names)} images have same dimensions: {reference_dim['height']}x{reference_dim['width']} ({reference_dim['bands']} bands)")
    
    # =========================================================================
    # STEP 2: Calculate patch grid and get validity mask for each month
    # =========================================================================
    h, w = reference_dim['height'], reference_dim['width']
    n_patches_h = int(np.ceil(h / PATCH_SIZE))
    n_patches_w = int(np.ceil(w / PATCH_SIZE))
    total_patches = n_patches_h * n_patches_w
    
    st.write(f"**Step 2: Patch grid**: {n_patches_h} x {n_patches_w} = **{total_patches} patches** per image")
    
    st.write("**Step 3: Calculating validity for each month...**")
    
    progress_bar = st.progress(0)
    
    # Store validity mask for each month
    month_validity_masks = {}
    month_valid_counts = {}
    
    for idx, month_name in enumerate(month_names):
        image_path = downloaded_images[month_name]
        validity_mask, orig_size, grid_size = get_patch_validity_mask(
            image_path, PATCH_SIZE, nodata_threshold_percent
        )
        
        if validity_mask is None:
            st.warning(f"⚠️ Could not analyze {month_name}")
            progress_bar.progress((idx + 1) / len(month_names))
            continue
        
        # Verify grid size matches expected
        if validity_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch! Expected {(n_patches_h, n_patches_w)}, got {validity_mask.shape}")
            return None, None, None
        
        month_validity_masks[month_name] = validity_mask
        month_valid_counts[month_name] = np.sum(validity_mask)
        
        progress_bar.progress((idx + 1) / len(month_names))
    
    progress_bar.empty()
    
    if len(month_validity_masks) == 0:
        st.error("❌ Could not analyze any months!")
        return None, None, None
    
    # =========================================================================
    # STEP 4: Find the month with MAXIMUM valid patches (reference)
    # =========================================================================
    st.write("**Step 4: Finding reference mask (maximum valid patches)...**")
    
    # Find max valid patch count
    max_valid_count = max(month_valid_counts.values())
    
    # Find all months with max valid count (could be multiple)
    max_months = [mn for mn, count in month_valid_counts.items() if count == max_valid_count]
    
    st.info(f"📊 Maximum valid patches: **{max_valid_count}/{total_patches}** ({100*max_valid_count/total_patches:.1f}%)")
    st.write(f"   Months with max patches: {', '.join(max_months)}")
    
    # Use the first max month as reference
    reference_month = max_months[0]
    reference_mask = month_validity_masks[reference_month]
    
    # =========================================================================
    # STEP 5: Check which months have ALL reference patches valid
    # =========================================================================
    st.write("**Step 5: Filtering months that match reference mask...**")
    
    valid_months = {}
    excluded_months = {}
    
    for month_name, mask in month_validity_masks.items():
        # Check if this month has ALL patches that are valid in reference
        # A month is valid if: everywhere reference_mask is True, this mask is also True
        matches_reference = np.all(mask[reference_mask] == True)
        
        if matches_reference:
            valid_months[month_name] = downloaded_images[month_name]
            st.write(f"   ✅ {month_name}: {month_valid_counts[month_name]}/{total_patches} patches - **INCLUDED**")
        else:
            # Count how many reference patches are missing
            missing_count = np.sum(reference_mask & ~mask)
            excluded_months[month_name] = {
                'valid_count': month_valid_counts[month_name],
                'missing_count': missing_count
            }
            st.write(f"   ❌ {month_name}: {month_valid_counts[month_name]}/{total_patches} patches - **EXCLUDED** (missing {missing_count} patches)")
    
    # =========================================================================
    # STEP 6: Summary
    # =========================================================================
    st.divider()
    
    if len(valid_months) == 0:
        st.error("❌ No months match the reference mask!")
        st.warning("This shouldn't happen - at least the reference month should match.")
        return None, None, None
    
    st.success(f"✅ **{len(valid_months)}/{len(month_validity_masks)}** months have all {max_valid_count} valid patches")
    
    if excluded_months:
        with st.expander(f"🚫 Excluded Months ({len(excluded_months)})", expanded=True):
            for month_name, info in excluded_months.items():
                st.write(f"  • {month_name}: had {info['valid_count']} patches, missing {info['missing_count']} from reference")
    
    original_size = (h, w)
    
    return reference_mask, original_size, valid_months


def classify_image_with_mask(image_path, model, device, month_name, valid_mask, original_size):
    """
    Classify an image, only processing valid patches.
    
    IMPORTANT: Validates that image dimensions match expected original_size
    and that patch grid matches valid_mask shape.
    
    Returns: (binary_mask, probability_map, valid_count)
    """
    try:
        with rasterio.open(image_path) as src:
            img_data = src.read()
        
        img_for_patching = np.moveaxis(img_data, 0, -1)
        
        h, w, c = img_for_patching.shape
        
        # VALIDATION: Check image dimensions match expected size
        if original_size is not None:
            expected_h, expected_w = original_size
            if h != expected_h or w != expected_w:
                st.error(f"❌ {month_name}: Dimension mismatch! Expected {expected_h}x{expected_w}, got {h}x{w}")
                return None, None, 0
        
        new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
        new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)
        
        # Calculate expected patch grid
        n_patches_h = new_h // PATCH_SIZE
        n_patches_w = new_w // PATCH_SIZE
        
        # VALIDATION: Check patch grid matches valid_mask
        if valid_mask.shape != (n_patches_h, n_patches_w):
            st.error(f"❌ {month_name}: Patch grid mismatch! Mask is {valid_mask.shape}, image needs {(n_patches_h, n_patches_w)}")
            return None, None, 0
        
        if h != new_h or w != new_w:
            padded_img = np.zeros((new_h, new_w, c), dtype=img_for_patching.dtype)
            padded_img[:h, :w, :] = img_for_patching
            img_for_patching = padded_img
        
        patches = patchify(img_for_patching, (PATCH_SIZE, PATCH_SIZE, c), step=PATCH_SIZE)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        
        classified_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        probability_patches = np.zeros((n_patches_h, n_patches_w, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        
        valid_count = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if not valid_mask[i, j]:
                    continue
                
                patch = patches[i, j, 0]
                patch_normalized = normalized(patch)
                
                patch_tensor = torch.tensor(np.moveaxis(patch_normalized, -1, 0), dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0)
                
                with torch.inference_mode():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction).cpu()
                
                pred_np = prediction.squeeze().numpy()
                classified_patches[i, j] = (pred_np > 0.5).astype(np.uint8) * 255
                probability_patches[i, j] = pred_np  # Store probability values
                valid_count += 1
        
        reconstructed = unpatchify(classified_patches, (new_h, new_w))
        reconstructed = reconstructed[:original_size[0], :original_size[1]]
        
        reconstructed_prob = unpatchify(probability_patches, (new_h, new_w))
        reconstructed_prob = reconstructed_prob[:original_size[0], :original_size[1]]
        
        return reconstructed, reconstructed_prob, valid_count
    except:
        return None, None, 0


def get_valid_patch_bounds(valid_mask, patch_size=224, original_size=None):
    """
    Calculate the pixel bounds of valid patches.
    Returns: (row_start, row_end, col_start, col_end) in pixels
    """
    if valid_mask is None or not np.any(valid_mask):
        return None
    
    # Find rows and cols with valid patches
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)
    
    # Get min/max patch indices
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None
    
    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]
    
    # Convert to pixel coordinates
    row_start = min_row * patch_size
    row_end = (max_row + 1) * patch_size
    col_start = min_col * patch_size
    col_end = (max_col + 1) * patch_size
    
    # Clip to original size if provided
    if original_size is not None:
        row_end = min(row_end, original_size[0])
        col_end = min(col_end, original_size[1])
    
    return (row_start, row_end, col_start, col_end)


def create_pixel_mask_from_patches(valid_mask, patch_size=224, target_size=None):
    """
    Create a pixel-level mask from patch-level validity mask.
    Each patch's validity is expanded to cover all its pixels.
    
    Returns: 2D boolean array of shape target_size (or expanded patch grid size)
    """
    if valid_mask is None:
        return None
    
    n_patches_h, n_patches_w = valid_mask.shape
    
    # Create pixel mask by repeating each patch value
    pixel_mask = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    
    # Crop to target size if provided
    if target_size is not None:
        pixel_mask = pixel_mask[:target_size[0], :target_size[1]]
    
    return pixel_mask


def generate_thumbnails(image_path, classification_mask, month_name, valid_mask=None, original_size=None, max_size=256):
    """
    Generate both RGB and classification thumbnails.
    If valid_mask is provided:
    1. Crops both images to the valid patch bounding box
    2. Masks out invalid patches in RGB (shows black where patches are invalid)
    This ensures RGB and classification show exactly the same valid regions.
    """
    try:
        # Get crop bounds and pixel mask from valid_mask
        crop_bounds = None
        pixel_mask = None
        if valid_mask is not None:
            crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size)
            pixel_mask = create_pixel_mask_from_patches(valid_mask, PATCH_SIZE, original_size)
        
        # Read and process RGB
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                p_low = np.percentile(valid, lower)
                p_high = np.percentile(valid, upper)
                if p_high <= p_low:
                    p_high = p_low + 0.001
                stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                return (stretched * 255).astype(np.uint8)
            
            rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
        
        # Apply pixel mask to RGB (set invalid patches to black)
        if pixel_mask is not None:
            # Expand mask to 3 channels
            pixel_mask_3ch = np.stack([pixel_mask, pixel_mask, pixel_mask], axis=-1)
            rgb_uint8 = np.where(pixel_mask_3ch, rgb_uint8, 0)
        
        # Crop both images if bounds are available
        if crop_bounds is not None:
            row_start, row_end, col_start, col_end = crop_bounds
            rgb_cropped = rgb_uint8[row_start:row_end, col_start:col_end, :]
            class_cropped = classification_mask[row_start:row_end, col_start:col_end]
        else:
            rgb_cropped = rgb_uint8
            class_cropped = classification_mask
        
        # Create PIL images
        pil_rgb = Image.fromarray(rgb_cropped, mode='RGB')
        pil_class = Image.fromarray(class_cropped.astype(np.uint8))
        
        # Resize if needed
        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_class = pil_class.resize((new_w, new_h), Image.NEAREST)
        
        # Calculate building stats from cropped region (only valid pixels)
        building_pixels = np.sum(class_cropped > 0)
        total_pixels = class_cropped.shape[0] * class_cropped.shape[1]
        
        return {
            'rgb_image': pil_rgb,
            'classification_image': pil_class,
            'month_name': month_name,
            'original_size': classification_mask.shape,
            'cropped_size': class_cropped.shape,
            'building_pixels': building_pixels,
            'total_pixels': total_pixels
        }
    except Exception as e:
        st.warning(f"Error generating thumbnails for {month_name}: {e}")
        return None


# =============================================================================
# MAIN PROCESSING PIPELINE WITH RESUME CAPABILITY
# =============================================================================
def process_timeseries(aoi, start_date, end_date, model, device,
                       cloudy_pixel_percentage=10, scale=10, nodata_threshold_percent=5,
                       resume=False):
    """Main processing pipeline - v05 style: combined analysis + download."""
    try:
        if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
            st.session_state.current_temp_dir = tempfile.mkdtemp()
        temp_dir = st.session_state.current_temp_dir
        
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        st.info(f"📅 Processing {total_months} months | 📁 {temp_dir}")
        
        # Extended date range (M-1 to M+1) for gap-filling
        extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')
        
        # =====================================================================
        # PHASE 1: Create cloud-free collection
        # =====================================================================
        st.header("Phase 1: Cloud-Free Collection")
        st.info(f"☁️ Cloud mask: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}")
        
        cloud_free_collection = create_cloud_free_collection(
            aoi, extended_start, extended_end, cloudy_pixel_percentage
        )
        
        # =====================================================================
        # PHASE 2: Download all months (v05 style - combined analysis + download)
        # =====================================================================
        st.header("Phase 2: Download & Gap-Fill")
        
        downloaded_images = {}
        month_statuses = {}
        
        # Prepare month info list FIRST - so we know which months are in current range
        month_infos = []
        current_range_months = set()
        for month_index in range(total_months):
            # Proper month calculation: add months to start date
            year = start_dt.year + (start_dt.month - 1 + month_index) // 12
            month = (start_dt.month - 1 + month_index) % 12 + 1
            month_name = f"{year}-{month:02d}"
            month_infos.append({
                'month_name': month_name,
                'month_index': month_index,
                'origin': start_date
            })
            current_range_months.add(month_name)
        
        # Show expected months for clarity
        st.info(f"📅 Expected months: {month_infos[0]['month_name']} to {month_infos[-1]['month_name']} ({len(month_infos)} months, end date is EXCLUSIVE)")
        
        # Check existing cached downloads (ONLY for months in current range)
        if resume and st.session_state.downloaded_images:
            for month_name, path in st.session_state.downloaded_images.items():
                # Only consider months in current date range
                if month_name not in current_range_months:
                    continue
                    
                if os.path.exists(path):
                    is_valid, _ = validate_geotiff_file(path, len(SPECTRAL_BANDS))
                    if is_valid:
                        downloaded_images[month_name] = path
                        month_statuses[month_name] = {'status': STATUS_COMPLETE, 'message': 'Cached'}
            
            if downloaded_images:
                st.info(f"🔄 Found {len(downloaded_images)} cached downloads")
                for mn in sorted(downloaded_images.keys()):
                    st.write(f"🟢 **{mn}**: complete (cached)")
        
        # Restore previous month statuses (ONLY for months in current range that aren't downloaded)
        if resume and st.session_state.month_analysis_results:
            for month_name, status_info in st.session_state.month_analysis_results.items():
                # Only consider months in current date range
                if month_name not in current_range_months:
                    continue
                    
                if month_name not in month_statuses:
                    month_statuses[month_name] = status_info
                    status = status_info.get('status', 'unknown')
                    message = status_info.get('message', '')
                    icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                    st.write(f"{icon} **{month_name}**: {status} (cached) - {message}")
        
        # Process remaining months (not in downloaded_images AND not already processed with status)
        months_to_process = [m for m in month_infos 
                            if m['month_name'] not in downloaded_images 
                            and m['month_name'] not in month_statuses]
        
        if months_to_process:
            st.info(f"📥 {len(months_to_process)} months to process: {', '.join([m['month_name'] for m in months_to_process])}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, month_info in enumerate(months_to_process):
                month_name = month_info['month_name']
                
                # Safeguard: skip if already processed (shouldn't happen, but just in case)
                if month_name in month_statuses:
                    st.warning(f"⚠️ {month_name} already processed, skipping...")
                    progress_bar.progress((idx + 1) / len(months_to_process))
                    continue
                
                # Download with v06 cloud masking + gap-filling (all in one function)
                path, status, message = download_monthly_image_v06(
                    aoi=aoi,
                    cloud_free_collection=cloud_free_collection,
                    month_info=month_info,
                    temp_dir=temp_dir,
                    scale=scale,
                    status_placeholder=status_text
                )
                
                # Update both local and session state immediately
                month_statuses[month_name] = {'status': status, 'message': message}
                st.session_state.month_analysis_results[month_name] = {'status': status, 'message': message}
                
                icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
                st.write(f"{icon} **{month_name}**: {status} - {message}")
                
                if path:
                    downloaded_images[month_name] = path
                    st.session_state.downloaded_images[month_name] = path
                
                progress_bar.progress((idx + 1) / len(months_to_process))
            
            progress_bar.empty()
            status_text.empty()
        
        # Summary
        st.divider()
        status_counts = {s: sum(1 for ms in month_statuses.values() if ms['status'] == s) 
                         for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Complete", status_counts[STATUS_COMPLETE])
        col2.metric("🔴 Rejected", status_counts[STATUS_REJECTED])
        col3.metric("🟡 Skipped", status_counts[STATUS_SKIPPED])
        col4.metric("⚫ No Data", status_counts[STATUS_NO_DATA])
        
        # Update session state
        failed_downloads = [mn for mn, ms in month_statuses.items() if ms['status'] != STATUS_COMPLETE]
        st.session_state.failed_downloads = failed_downloads
        st.session_state.month_analysis_results = month_statuses
        
        if not downloaded_images:
            st.error("❌ No images downloaded!")
            return []
        
        st.success(f"✅ Downloaded {len(downloaded_images)}/{total_months} months")
        
        # =====================================================================
        # PHASE 3: Patch validity - Find MAX patches and exclude non-conforming months
        # =====================================================================
        st.header("Phase 3: Patch Validity Analysis")
        
        valid_mask, original_size, valid_months = find_common_valid_patches(downloaded_images, nodata_threshold_percent)
        
        if valid_mask is None or valid_months is None:
            return []
        
        st.session_state.valid_patches_mask = valid_mask
        st.session_state.valid_months = valid_months  # Save for PDF report generation
        
        # IMPORTANT: Clear processed_months cache when valid_mask changes
        # This ensures all thumbnails are generated with the same mask
        if st.session_state.processed_months:
            st.info("🔄 Clearing thumbnail cache to ensure consistency...")
        st.session_state.processed_months = {}
        
        # Show which months were excluded
        excluded_count = len(downloaded_images) - len(valid_months)
        if excluded_count > 0:
            st.warning(f"⚠️ {excluded_count} months excluded due to missing patches")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(valid_mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Reference Valid Patches ({np.sum(valid_mask)} patches)")
        st.pyplot(fig)
        plt.close()
        
        # =====================================================================
        # PHASE 4: Classification (only valid months, with disk cache)
        # =====================================================================
        st.header("Phase 4: Classification")
        
        st.info(f"🧠 Classifying **{len(valid_months)}** months (excluded {excluded_count} with missing patches)")
        
        # Create cache directory for classification results
        classification_cache_dir = os.path.join(temp_dir, "classification_cache")
        os.makedirs(classification_cache_dir, exist_ok=True)
        
        thumbnails = []
        probability_maps = {}  # Store probability maps for change detection
        month_names = sorted(valid_months.keys())  # Use valid_months, not downloaded_images
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, month_name in enumerate(month_names):
            # Define cache file paths for this month
            prob_map_cache = os.path.join(classification_cache_dir, f"{month_name}_probmap.npy")
            thumbnail_cache = os.path.join(classification_cache_dir, f"{month_name}_thumbnail.pkl")
            
            # Check DISK cache first
            if os.path.exists(prob_map_cache) and os.path.exists(thumbnail_cache):
                try:
                    # Load from disk
                    prob_map = np.load(prob_map_cache)
                    import pickle
                    with open(thumbnail_cache, 'rb') as f:
                        thumb = pickle.load(f)
                    
                    # Add to results
                    probability_maps[month_name] = prob_map
                    st.session_state.probability_maps[month_name] = prob_map
                    thumbnails.append(thumb)
                    st.session_state.processed_months[month_name] = thumb
                    
                    progress_bar.progress((idx + 1) / len(month_names))
                    continue  # Skip classification, use cached result
                except Exception as e:
                    # Cache corrupted, re-classify
                    st.warning(f"⚠️ {month_name}: Cache corrupted ({e}), re-classifying...")
            
            # Otherwise, classify this month
            status_text.text(f"🧠 {month_name} ({idx+1}/{len(month_names)})...")
            
            mask, prob_map, valid_count = classify_image_with_mask(
                valid_months[month_name], model, device, month_name, valid_mask, original_size
            )
            
            if mask is not None:
                # Store probability map for change detection
                probability_maps[month_name] = prob_map
                st.session_state.probability_maps[month_name] = prob_map
                
                thumb = generate_thumbnails(
                    valid_months[month_name], mask, month_name,
                    valid_mask=valid_mask, original_size=original_size
                )
                if thumb:
                    thumb['valid_patches'] = valid_count
                    # Add gap-fill info from status
                    ms = month_statuses.get(month_name, {})
                    thumb['gap_filled'] = 'gap-fill' in ms.get('message', '').lower()
                    
                    thumbnails.append(thumb)
                    st.session_state.processed_months[month_name] = thumb
                    
                    # Save to DISK cache
                    try:
                        np.save(prob_map_cache, prob_map)
                        import pickle
                        with open(thumbnail_cache, 'wb') as f:
                            pickle.dump(thumb, f)
                    except Exception as e:
                        st.warning(f"⚠️ {month_name}: Failed to cache results ({e})")
            
            progress_bar.progress((idx + 1) / len(month_names))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Classified {len(thumbnails)} months!")
        
        # Filter out images with abnormally low building percentage (likely haze/snow)
        if len(thumbnails) >= 2:
            building_pcts = [(t['building_pixels'] / t['total_pixels']) * 100 for t in thumbnails]
            median_pct = np.median(building_pcts)
            
            filtered_thumbnails = []
            for t in thumbnails:
                pct = (t['building_pixels'] / t['total_pixels']) * 100
                if pct < (median_pct - 8.0):
                    print(f"🚫 Excluded {t['month_name']}: {pct:.1f}% buildings (median={median_pct:.1f}%) - likely haze or snow")
                    # Also remove from valid_months
                    if t['month_name'] in st.session_state.valid_months:
                        del st.session_state.valid_months[t['month_name']]
                    # Also remove from probability_maps for change detection
                    if t['month_name'] in st.session_state.probability_maps:
                        del st.session_state.probability_maps[t['month_name']]
                else:
                    filtered_thumbnails.append(t)
            
            thumbnails = filtered_thumbnails
        
        return thumbnails
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


# =============================================================================
# Change Detection Algorithm
# =============================================================================
def analyze_building_transition(probability_maps, non_building_thr=0.2, building_thr=0.8,
                                non_building_duration=2, building_duration=2):
    """
    Analyze building transition based on time series of probability maps.
    
    Algorithm:
    - Pixel must START as non-building (< non_building_thr)
    - Pixel must END as building (>= building_thr)
    - Only ONE transition from non-building to building allowed (no back-and-forth)
    - Must stay non-building for at least `non_building_duration` images before transition
    - Must stay building for at least `building_duration` images after transition
    
    Args:
        probability_maps: Dict of {month_name: probability_array}
        non_building_thr: Threshold below which pixel is non-building (default 0.2)
        building_thr: Threshold above which pixel is building (default 0.8)
        non_building_duration: Min months as non-building before transition (default 2)
        building_duration: Min months as building after transition (default 2)
    
    Returns:
        change_mask: Binary array where 1 = change detected
        stats: Dictionary with statistics
    """
    # Sort months chronologically
    sorted_months = sorted(probability_maps.keys())
    
    if len(sorted_months) < (non_building_duration + building_duration):
        return None, None, {"error": f"Need at least {non_building_duration + building_duration} months, got {len(sorted_months)}"}
    
    # Stack probability maps into 3D array (time, height, width)
    first_map = probability_maps[sorted_months[0]]
    height, width = first_map.shape
    n_times = len(sorted_months)
    
    data = np.zeros((n_times, height, width), dtype=np.float32)
    for i, month in enumerate(sorted_months):
        data[i] = probability_maps[month]
    
    # Initialize results
    results = np.zeros((height, width), dtype=np.uint8)
    # NEW: Track which month each pixel transitioned (0 = no transition, 1-N = month index)
    transition_timing = np.zeros((height, width), dtype=np.int16)
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            pixel_series = data[:, y, x]
            
            # Skip if pixel is all zeros (invalid/masked area)
            if np.all(pixel_series == 0):
                continue
            
            # CHECK 1: Pixel must start as non-building
            if pixel_series[0] > non_building_thr:
                continue
            
            # CHECK 2: Pixel must end as building
            if pixel_series[-1] < building_thr:
                continue
            
            # Find the transition point from non-building to building
            transition_point = None
            invalid_transition = False
            
            for t in range(1, len(pixel_series)):
                # Check for invalid backward transitions
                # If pixel was above non_building_thr and goes back below
                if pixel_series[t-1] > non_building_thr and pixel_series[t] <= non_building_thr:
                    invalid_transition = True
                    break
                # If pixel was above building_thr and goes back below
                elif pixel_series[t-1] >= building_thr and pixel_series[t] < building_thr:
                    invalid_transition = True
                    break
                
                # Detect the transition point (first time going from non-building to building)
                if (pixel_series[t-1] <= non_building_thr and 
                    pixel_series[t] >= building_thr and 
                    transition_point is None):
                    transition_point = t
            
            # Check if we found a valid transition
            if transition_point is not None and not invalid_transition:
                # Check duration constraints
                nb_duration = transition_point  # Duration before transition
                b_duration = len(pixel_series) - transition_point  # Duration after transition
                
                if (nb_duration >= non_building_duration and 
                    b_duration >= building_duration):
                    results[y, x] = 1
                    transition_timing[y, x] = transition_point  # Store the month index
    
    # Calculate statistics
    change_pixels = np.sum(results == 1)
    total_pixels = np.sum(data[0] > 0)  # Count non-zero pixels
    change_percentage = (change_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    # Calculate per-month transition statistics
    transition_by_month = {}
    for i, month in enumerate(sorted_months):
        if i > 0:  # Skip first month (can't transition TO first month)
            count = np.sum(transition_timing == i)
            transition_by_month[month] = int(count)
    
    stats = {
        'change_pixels': int(change_pixels),
        'total_pixels': int(total_pixels),
        'change_percentage': float(change_percentage),
        'n_months': n_times,
        'months': sorted_months,
        'first_month': sorted_months[0],
        'last_month': sorted_months[-1],
        'transition_by_month': transition_by_month
    }
    
    # Store timing map in session state for later use
    st.session_state.change_timing_map = transition_timing
    
    return results, transition_timing, stats


def generate_rgb_from_sentinel(image_path, max_size=None):
    """Generate RGB image from Sentinel-2 bands for display."""
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            rgb = np.stack([red, green, blue], axis=-1)
            rgb = np.nan_to_num(rgb, nan=0.0)
            
            def percentile_stretch(band, lower=2, upper=98):
                valid = band[band > 0]
                if len(valid) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                p_low = np.percentile(valid, lower)
                p_high = np.percentile(valid, upper)
                if p_high <= p_low:
                    p_high = p_low + 0.001
                stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                return (stretched * 255).astype(np.uint8)
            
            rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])
            
            return rgb_uint8
    except:
        return None


# =============================================================================
# Display Thumbnails
# =============================================================================
def get_image_download_data(image_path, month_name):
    """Read GeoTIFF file and return bytes for download."""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        st.warning(f"Error reading {month_name}: {e}")
        return None


def get_classification_geotiff_wgs84(prob_map, source_image_path, month_name):
    """
    Build a georeferenced classification GeoTIFF (binary building mask),
    reprojected to WGS84 (EPSG:4326) using the source Sentinel-2 image's
    CRS/transform, and return the file bytes for download.
    """
    try:
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        with rasterio.open(source_image_path) as src:
            src_crs = src.crs
            src_transform = src.transform

        # Binary building mask (0 / 255) at full source resolution
        class_mask = (prob_map > 0.5).astype(np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmp_dir:
            native_path = os.path.join(tmp_dir, f"{month_name}_native.tif")
            wgs84_path = os.path.join(tmp_dir, f"{month_name}_wgs84.tif")

            # Write classification mask in the source raster's native CRS
            with rasterio.open(
                native_path, 'w', driver='GTiff',
                height=class_mask.shape[0], width=class_mask.shape[1],
                count=1, dtype='uint8', crs=src_crs, transform=src_transform
            ) as dst:
                dst.write(class_mask, 1)

            # Reproject to WGS84 (EPSG:4326)
            with rasterio.open(native_path) as src_native:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_native.crs, 'EPSG:4326', src_native.width, src_native.height, *src_native.bounds
                )
                dst_kwargs = src_native.meta.copy()
                dst_kwargs.update({
                    'crs': 'EPSG:4326',
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })

                with rasterio.open(wgs84_path, 'w', **dst_kwargs) as dst_wgs:
                    reproject(
                        source=rasterio.band(src_native, 1),
                        destination=rasterio.band(dst_wgs, 1),
                        src_transform=src_native.transform,
                        src_crs=src_native.crs,
                        dst_transform=dst_transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.nearest
                    )

            with open(wgs84_path, 'rb') as f:
                return f.read()
    except Exception as e:
        st.warning(f"Error preparing georeferenced classification image for {month_name}: {e}")
        return None


# =============================================================================
# Persian presentation layer (scientific functions above are preserved verbatim)
# =============================================================================

WQ_VAZIRMATN_OFL = 'Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn)\n\nThis Font Software is licensed under the SIL Open Font License, Version 1.1.\nThis license is copied below, and is also available with a FAQ at:\nhttp://scripts.sil.org/OFL\n\n\n-----------------------------------------------------------\nSIL OPEN FONT LICENSE Version 1.1 - 26 February 2007\n-----------------------------------------------------------\n\nPREAMBLE\nThe goals of the Open Font License (OFL) are to stimulate worldwide\ndevelopment of collaborative font projects, to support the font creation\nefforts of academic and linguistic communities, and to provide a free and\nopen framework in which fonts may be shared and improved in partnership\nwith others.\n\nThe OFL allows the licensed fonts to be used, studied, modified and\nredistributed freely as long as they are not sold by themselves. The\nfonts, including any derivative works, can be bundled, embedded, \nredistributed and/or sold with any software provided that any reserved\nnames are not used by derivative works. The fonts and derivatives,\nhowever, cannot be released under any other type of license. The\nrequirement for fonts to remain under this license does not apply\nto any document created using the fonts or their derivatives.\n\nDEFINITIONS\n"Font Software" refers to the set of files released by the Copyright\nHolder(s) under this license and clearly marked as such. This may\ninclude source files, build scripts and documentation.\n\n"Reserved Font Name" refers to any names specified as such after the\ncopyright statement(s).\n\n"Original Version" refers to the collection of Font Software components as\ndistributed by the Copyright Holder(s).\n\n"Modified Version" refers to any derivative made by adding to, deleting,\nor substituting -- in part or in whole -- any of the components of the\nOriginal Version, by changing formats or by porting the Font Software to a\nnew environment.\n\n"Author" refers to any designer, engineer, programmer, technical\nwriter or other person who contributed to the Font Software.\n\nPERMISSION & CONDITIONS\nPermission is hereby granted, free of charge, to any person obtaining\na copy of the Font Software, to use, study, copy, merge, embed, modify,\nredistribute, and sell modified and unmodified copies of the Font\nSoftware, subject to the following conditions:\n\n1) Neither the Font Software nor any of its individual components,\nin Original or Modified Versions, may be sold by itself.\n\n2) Original or Modified Versions of the Font Software may be bundled,\nredistributed and/or sold with any software, provided that each copy\ncontains the above copyright notice and this license. These can be\nincluded either as stand-alone text files, human-readable headers or\nin the appropriate machine-readable metadata fields within text or\nbinary files as long as those fields can be easily viewed by the user.\n\n3) No Modified Version of the Font Software may use the Reserved Font\nName(s) unless explicit written permission is granted by the corresponding\nCopyright Holder. This restriction only applies to the primary font name as\npresented to the users.\n\n4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font\nSoftware shall not be used to promote, endorse or advertise any\nModified Version, except to acknowledge the contribution(s) of the\nCopyright Holder(s) and the Author(s) or with their explicit written\npermission.\n\n5) The Font Software, modified or unmodified, in part or in whole,\nmust be distributed entirely under this license, and must not be\ndistributed under any other license. The requirement for fonts to\nremain under this license does not apply to any document created\nusing the Font Software.\n\nTERMINATION\nThis license becomes null and void if any of the above conditions are\nnot met.\n\nDISCLAIMER\nTHE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,\nEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF\nMERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT\nOF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE\nCOPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,\nINCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL\nDAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\nFROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM\nOTHER DEALINGS IN THE FONT SOFTWARE.\n'

WQ_VAZIRMATN_WOFF2_BASE64 = (
    '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'
    'AAA='
)

# Unmodified installed B Nazanin regular/bold; original copyright and
# font metadata are retained in the embedded bytes: (c) 2000 Borna Rayaneh.
BCD_BNAZANIN_REGULAR_TTF_BASE64 = (
    'AAEAAAAQAQAABAAAR1NVQifcOBsAAOa8AAAG+E9TLzK2Rt/mAAABiAAAAFZQQ0xUfEaxzgAA5oQAAAA2Y21hcPkyieIAABNYAAAD'
    'QmN2dCBpY8t1AAAXKAAAAHRmcGdtgzPCTwAAFpwAAAAUZ2x5ZghHoPMAABlYAADEmmhkbXh3Ey5bAAAFUAAADghoZWFk0/hvSgAA'
    'AQwAAAA2aGhlYQw1AosAAAFEAAAAJGhtdHil3UVQAAAB4AAAA25sb2NhZBqXKQAAF5wAAAG8bWF4cAGKAh0AAAFoAAAAIG5hbWXY'
    'm0jpAADd9AAAAnZwb3N0dvBTjwAA4GwAAAYWcHJlcLAxLGMAABawAAAAdQABAAAAAQAAFUK7SV8PPPUAAAgAAAAAALb4magAAAAA'
    'tviMFP8t++sHxgirAAAAAwACAAAAAAAAAAEAAAU//T8AAAer/y38xgfGAAEAAAAAAAAAAAAAAAAAAADaAAEAAADdAH4ACQBVAAMA'
    'AgAIAEAACgAAAIwBBgACAAIAAQAAAZAABQAABZoFMwAAASUFmgUzAAADoABmAhIAAAAABAAAAAAAAAAAAGAAgAAAAAAAAAgAAAAA'
    'TVo3MwBAACD+/AfG++sAAAfGBBUAAABAAAAAAAAABAAAgAAAAAAB9AAAAfQAAAFNAJoDfgBFAkAAnAJAAJsEFQCbAv8ANwGiAJ8C'
    '/wA3AU4AmwGnACsDwAEcA8ABEwPAAHYDwAA1A8AAagPAAGQDeABQA8AATAPAAEsDwACkAU4AmwL/ADcCgADWAoAAhwPVAR8D1QCW'
    'A1YARgNWADIC/wBJA7IAaQAlAAAAJQAAACUAAAAl/y0B6wB9AesAaQKKAEYCigBOAgAARAIAAEABogCcAaIAnAK8AJsCav/pAAAA'
    'ZQAAACsAAACZAAAAZQAAACwAAACYAAAAKgAAAIMAAAB7AAAARAAAAHgAAABpAAAAaAAAAC8AAABIAAAAOwAAAD0AAABiAAAAHAAA'
    'AEcAAABiAAD//gAAAEcAAAAyAAAATwAAADQAAABGAAAARgJ6AJ4B+gB2Al0AdgGrAIICHgCLApMAVAMAAFYBqwCmAhUAtwUuAJsE'
    '/QCbAU3/5gJ2/+YBLQCbAeAAmwVrAJsF5wCbAU3/5gJ2/+YFawCbBecAmwHM/+YCdv/mBWsAmwXnAJsBsf/mAnb/5gVrAJsF5wCb'
    'AbH/5gJ2/+YEJQCYBHkAmAQO/+YEX//mBCUAmAR5AJgEDv/mBF//5gQlAJgEeQCYBA7/5gRf/+YEJQCYBHkAmAQO/+YEX//mAw4A'
    'mgPLAJsDDgCaA8sAmwKHAEUDAgBFAocARQMCAEUChwBFAwIARQbwAJsHiwCbBBH/5gSc/+YG8ACbB4sAmwQR/+YEnP/mB1wAmwer'
    'AJsEd//mBMj/5gdcAJsHqwCbBHf/5gTI/+YFMQCeBYEAngPq/+YEO//mBTEAngWBAJ4D6v/mBDv/5gOYAJsEJQCbAsL/5gLP/+YD'
    'mACbBCUAmwLC/+YCz//mBWMAmgXRAJsCCf/mArT/5gSFAJsE7wCbAgn/5gK0/+YGVgCbBscAmwL9/+YDcP/mBlkAmwbHAJsC/f/m'
    'A2L/5gROAJsEzACbAUj/5gHZ/+YDMwCXBAcAlwKW/+YDlP/mBF4AmwTyAJsBTf/mAnb/5gKTAFQDAABWAoEAmwNSAJsDKP/mAxv/'
    '5gKBAJsDUgCbBS4AmwT9AJsBx//mAnb/5gPc/+AEkv/gA9wAGQSSABkD3ABTBJIATAPcAFMEkgBMBS4AmwT9AJsFiwCbAZUAVQAB'
    'AL8B0wCRAZkAAAAAABAAAADgCQkFAAICAQQDAwUDAgMBAgQEBAQEBAQEBAQBAwMDBAQEBAMEAAAAAAICAwMCAgICAwMAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAwIDAgIDAwICBgYBAwECBgcBAwYHAgMGBwIDBgcCAwUFBQUFBQUFBQUFBQUFBQUDBAMEAwMD'
    'AwMDCAgFBQgIBQUICQUFCAkFBQYGBAUGBgQFBAUDAwQFAwMGBwIDBQYCAwcIAwQHCAMEBQUBAgQFAwQFBgEDAwMDBAQDAwQGBgID'
    'BAUEBQQFBAUGBgYCAAAAAAAKCgUAAgICBAMDBQQCBAICBQUFBQUFBAUFBQIEAwMFBQQEBAUAAAAAAgIDAwMDAgIDAwAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAADAgMCAwMEAgMGBgIDAQIHBwIDBwcCAwcHAgMHBwIDBQYFBQUGBQUFBgUFBQYFBQQFBAUDBAME'
    'AwQJCQUGCQkFBgkKBgYJCgYGBgcFBQYHBQUEBQMEBAUDBAcHAwMGBgMDCAgEBAgIBAQFBgICBAUDBAUGAgMDBAMEBAQDBAYGAgMF'
    'BgUGBQYFBgYGBwIAAAAAAAsLBgADAwIFAwMGBAIEAgIFBQUFBQUFBQUFAgQDAwUFBQUEBQAAAAADAwMDAwMCAgQDAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAMDAwIDBAQCAwcHAgMCAwcIAgMHCAIDBwgCAwcIAgMGBgYGBgYGBgYGBgYGBgYGBAUEBQMEAwQD'
    'BAoKBgYKCgYGCgsGBwoLBgcHCAUGBwgFBgUGBAQFBgQEBwgDBAYHAwQJCQQFCQkEBQYHAgMEBgQFBgcCAwQEAwUEBAMFBwcCAwUG'
    'BQYFBgUGBwcIAgAAAAAADAwGAAMDAgUDAwYEAgQCAgYGBgYGBgUGBgYCBAQEBgYFBQQGAAAAAAMDBAQDAwICBAQAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAABAMEAwMEBQMDCAcCBAIDCAkCBAgJAwQICQMECAkDBAYHBgcGBwYHBgcGBwYHBgcFBgUGBAUEBQQF'
    'CgsGBwoLBgcLDAcHCwwHBwgIBgYICAYGBQYEBAUGBAQICQMEBwcDBAoKBAUKCgQFBgcCAwUGBAUHBwIEBAUEBQUFBAUIBwMEBgcG'
    'BwYHBgcIBwgCAAAAAAANDAcAAwMCBgQEBwUDBQIDBgYGBgYGBgYGBgIFBAQGBgUFBQYAAAAAAwMEBAMDAwMEBAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAEAwQDAwQFAwMICAIEAgMJCgIECQoDBAkKAwQJCgMEBwcHBwcHBwcHBwcHBwcHBwUGBQYEBQQFBAUL'
    'DAcHCwwHBwwMBwgMDAcICAkGBwgJBgcGBwQFBgcEBQkJAwQHCAMECgsFBgoLBQUHCAIDBQcEBgcIAgQEBQQFBQUEBQgIAwQGBwYH'
    'BgcGBwgICQMAAAAAAA4NBwADAwIGBAQHBQMFAgMHBwcHBwcGBwcHAgUEBAcHBgYFBgAAAAADAwQEBAQDAwUEAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAQDBAMEBQUDBAkJAgQCAwkKAgQJCgMECQoDBAkKAwQHCAcIBwgHCAcIBwgHCAcIBQcFBwQFBAUEBQwN'
    'BwgMDQcIDQ0ICA0NCAgJCgcHCQoHBwYHBQUGBwUFCQoEBQgJBAULDAUGCwwFBggIAgMGBwUGCAkCBAUFBAYGBQQGCQkDBAcIBwgH'
    'CAcICQkKAwAAAAAADw4IAAQEAgcEBAgGAwYCAwcHBwcHBwcHBwcCBgUFBwcGBgYHAAAAAAQEBQUEBAMDBQUAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAABQQEAwQFBgMECgkCBQIECgsCBQoLAwUKCwMFCgsDBQgICAgICAgICAgICAgICAgGBwYHBQYFBgUGDQ4I'
    'CQ0OCAkODggJDg4ICQoKBwgKCgcIBwgFBQcIBQUKCwQFCAkEBQwNBgYMDQYGCAkCAwYIBQcICQIFBQYFBgYGBQYKCQMFBwkHCQcJ'
    'BwkKCQoDAAAAAAAQDwgABAQDBwUFCAYDBgMDCAgICAgIBwgICAMGBQUICAcHBgcAAAAABAQFBQQEAwMFBQAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAFBAUDBAUGAwQKCgMFAgQLDAMFCwwEBQsMAwULDAMFCAkICQgJCAkICQgJCAkICQYIBggFBgUGBQYODwgJ'
    'Dg8ICQ8PCQoPDwkKCgsICAoLCAgHCAYGBwgGBgsMBAUJCgQFDQ4GBw0OBgcJCgMEBggFBwkKAwUFBgUHBgYFBwoKBAUICQgJCAkI'
    'CQoKCwMAAAAAABEQCQAEBAMHBQUJBgMGAwQICAgICAgHCAgIAwYFBQgIBwcGCAAAAAAEBAUFBAQDAwYFAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAUEBQQEBQYEBAsLAwUCBAwNAwUMDQQFDA0EBQwNBAUJCgkJCQoJCQkKCQkJCgkJBggGCAUGBQYFBg8QCQoP'
    'EAkKEBAJChAQCQoLDAgJCwwICQgJBgYICQYGCwwEBgoKBAYNDgYHDQ4GBwkKAwQHCQUICQsDBQUGBQcHBwUHCwsEBQgKCAoICggK'
    'CwsMAwAAAAAAEhEJAAQEAwgFBQkHBAcDBAgICAgICAgICAgDBwYGCQkICAcIAAAAAAQEBgYFBQQEBgUAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAABgQFBAUGBwQFDAsDBgMEDA0DBgwNBAYMDQQGDA0EBgkKCQoJCgkKCQoJCgkKCQoHCQcJBgcGBwYHEBEJChAR'
    'CQoREQoLEREKCwwMCQoMDAkKCAkGBggJBgYMDQUGCgsFBg4PBwgODwcICgsDBAcJBggKCwMGBgcGBwcHBgcMCwQGCQoJCgkKCQoM'
    'CwwEAAAAAAATEgoABQUDCAUFCgcEBwMECQkJCQkJCAkJCQMHBgYJCQgIBwkAAAAABQUGBgUFBAQGBgAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAGBQYEBQYHBAUMDAMGAwQNDgMGDQ4EBg0OBAYNDgQGCgsKCgoLCgoKCwoKCgsKCgcJBwkGBwYHBgcQEgoLEBIK'
    'CxESCwsREgsLDA0JCgwNCQoJCgcHCQoHBw0OBQYLDAUGDxAHCA8QBwgKCwMECAoGCAoMAwYGBwYIBwcGCAwMBAYJCwkLCQsJCwwM'
    'DQQAAAAAABQTCgAFBQMJBgYKBwQHAwQJCQkJCQkJCQkJAwcGBgoKCAgHCQAAAAAFBQYGBQUEBAcGAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAYFBgQFBggEBQ0MAwYDBQ4PAwYODwQGDg8EBg4PBAYKCwoLCgsKCwoLCgsKCwoLCAkICQYIBggGCBETCgwREwoM'
    'EhMLDBITCwwNDgoLDQ4KCwkKBwcJCgcHDQ8FBwsMBQcQEQcJEBEHCAsMAwUICgYJCwwDBgYIBggICAYIDQwEBgoLCgsKCwoLDQwO'
    'BAAAAAAAFRQLAAUFAwkGBgsIBAgDBAoKCgoKCgkKCgoDCAcHCgoJCQgKAAAAAAUFBwcFBQQEBwYAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAABwUGBAYHCAQFDg0DBgMFDg8DBg4PBQYODwQGDg8EBgsMCwsLDAsLCwwLCwsMCwsICggKBwgHCAcIEhQLDBIUCwwT'
    'FAwNExQMDQ4OCgsODgoLCQsHBwkLBwcODwUHDA0FBxESCAkREggJCw0DBQgLBwkLDQMGBwgHCQgIBwkODQUGCgwKDAoMCgwODQ8E'
    'AAAAAAAWFQsABQUECgYGCwgECAQFCgoKCgoKCgoKCgQIBwcLCwkJCAoAAAAABQUHBwYGBAQIBwAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAHBQYFBgcIBQYODgQHAwUPEAQHDxAFBw8QBQcPEAUHCwwLDAsMCwwLDAsMCwwLDAgKCAoHCAcIBwgTFQsNExULDRQV'
    'DA0UFQwNDg8LDA4PCwwKCwgICgsICA8QBgcMDgYHERMICRETCAkMDQQFCQsHCgwOBAcHCAcJCQkHCQ4OBQcLDQsNCw0LDQ4ODwQA'
    'AAAAABcWDAAGBgQKBgYMCQUJBAULCwsLCwsKCwsLBAkHBwsLCgoJCwAAAAAGBgcHBgYFBQgHAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAcGBwUGBwkFBg8OBAcDBRARBAcQEQUHEBEFBxARBQcMDQwNDA0MDQwNDA0MDQwNCQsJCwcJBwkHCRQWDA0UFgwNFRYN'
    'DhUWDQ4PEAsMDxALDAoMCAgKDAgIDxEGCA0OBggSEwkKEhMJCgwOBAUJDAcKDQ4EBwcJBwoJCQcKDw4FBwsNCw0LDQsNDw4QBQAA'
    'AAAAGBcMAAYGBAoHBwwJBQkEBQsLCwsLCwoLCwsECQgICwsKCgkLAAAAAAYGCAgGBgUFCAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAABwYHBQYICQUGEA8EBwQGEBIEBxASBQcQEgUHEBIFBwwNDA0MDQwNDA0MDQwNDA0JCwkLCAkICQgJFRcMDhUXDA4WFw0O'
    'FhcNDhARDA0QEQwNCwwICAsMCAgQEQYIDg8GCBMUCQoTFAkKDQ4EBgoMCAsNDwQHCAkICgkJCAoQDwUHDA4MDgwODA4QDxEFAAAA'
    'AAAAAAACAAEAAAAAABQAAwABAAABGgAAAQYAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwQAAAAFAAAGBwgJCgsM'
    'DQ4PEBESExQVFhcYAAAZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaABsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'ABwAHQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB4AAAAAAAAAAAAAANgAAAAfAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhAAAAAAAAAAAABAIoAAAAagBAAAUAKgAh'
    'ACUAOgA9AFsAXQB7AH0AqwC3ALsA1wD3AtkGDAYbBh8GOgZSBmkGfgaGBpgGqQavBsAGzAb5IA8gGSAdIDoiGegY6C37Wft9+4v7'
    'lful+//8Yv3y/pL+lP6g/rD+3P7o/uz+8P78//8AAAAgACUAKAA9AFsAXQB7AH0AqwC3ALsA1wD3AtkGDAYbBh8GIQZABmAGfgaG'
    'BpgGqQavBsAGzAbwIAwgGCAcIDkiGegY6CD7Vvt6+4r7jvuk+/z8Xv3y/oH+k/6V/qH+sf7d/un+7f7x////4//g/97/3P+//77/'
    'of+g/3MAIf9k/0n/Kv3/+iD6EvoPAAAAAPmu+eH56fnr+gT6AvoH+f35HuAW4A7gDN/x3r8YMBgYBQkE9QT5BR8FIwTNAAAC5QHM'
    'AAABzgHSAdQB2AHaAAAB2AABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABIAHoAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAByAAAAAAB2AAAAAAAAAAAAAABuAAAAAABMAE0ATwBRAFMAVQBZAFsAxwBjAGcAawBz'
    'AHcAewB9AH8AgQCFAIkAjQCRAJUAmQCdAKEALwClAKkArQC1ALkAvQDDAMEA1QDJADAAMQAyADMANAA1ADYANwBJAEsARgBHAEoA'
    'xwDIAMEAwgDVANYAAEABACx2RSCwAyVFI2FoGCNoYEQtQCMUFBMTEhIRERAQDw8ODg0NDAwLCwoKCQkICAcHBgYFBQAAAY24Af+F'
    'RWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhERWhEswIBRgArswQDRgArsQEBRWhEsQMDRWhE'
    'AAAAAAAAXQB5AKkA4wCsAFEBdQE/AvMDNAPzAg0DhgCuAlEDAwLWA3gDNAPrX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9p'
    'X2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaV9pX2lfaQAjACUAAAA/AD8APwA/AJYA8QE6AYICOQKEAscC9AMmA1oD'
    'mAPSBCMElwUPBXsF3wY0BogG5QcuB2wHqgfnCFMIvwkECUgJsAoLCgsKCwpeCq8K+As+C5wL+wwtDF4Mowz9DXoNpw33DmoOuQ7s'
    'D0kPfA/cECIQKxA0ED0QRhBPEFgQYRBqEHYRHBGyEb4SMxLFEs4S1xLgEukS8hL7E1cTuRQmFJ8VJxXMFn0W9Bd6GDUY6BlnGeoa'
    'BhpMGsEbPhueHAUcux15HhoewR9XH/UgdSD6IbEicCMRI7ckSST1JXsmGibsJ9konyl+Ke4qeSrdK1or6yyWLRstuS4MLm8u4y9n'
    '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'
    'lSYllpUmJZYABACb/fcG7wQLAEwAUABUAFgA90CAAVlZQFoAWFZUUk5JLxgIV1VTUU9NRiwoGk1QTU4FTk9QUE1PT1BTUlNUBVRR'
    'UlJTUVFSVVhVVgVWV1hYVVdXWE5NTk8FT1BNTU5QUE1SUVJTBVNUUVFSVFRRVlVWVwVXWFVVVlhYVR4BEkQBAEADAjUDBiIDDjsE'
    'BFAOAgABEkZ2LzcYAD8vLy/9EP0v/RD9AS/9L/2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8'
    'CMQBLi4uLi4uLi4uLgAuLi4uLi4uLi4xMAFJaLkAEgBZSWhhsEBSWDgRN7kAWf/AOFkBECMiJwYjIicGBwYHBiMiJyY1NDc2NzYz'
    'MhUUBwYVFBcWMzI3Njc2JyYnJic3NjMyFxYXFjMyNzY3NjMyFx4BMzI3NjU0Jzc2NxYXFgEHJzcTByc3DwEnNwbv5H4uP3gsMwQK'
    'M6mRqahsdisfIRANDAY5P1Oovo10HgoHBwwXZTgMBgUNNRQ2TFUmGQ8HEQ8EDFhgKB0hSkEKBQYKP/7JXV9e0V5eXYVdX14BDf7x'
    'cHMQc4GPTkNTXKRXY0coEw4JDHVhX0VbRDhCFj9AHzyCahcTUBM1Lh9OJiJdRhUXJjdTbxEBAw9hAjNeX13+2l5fXV5eX10AAAQA'
    'm/33B6UDqgBLAE8AUwBXAP9AhAFYWEBZAFdVU1FNMhsLVlRSUE5MSy8rHQBMT0xNBU1OT09MTk5PUFNQUQVRUlNTUFJSU1RXVFUF'
    'VVZXV1RWVldNTE1OBU5PTExNT09MUVBRUgVSU1BQUVNTUFVUVVYFVldUVFVXV1QhARVLSgMAOEIDCQUlAxEDBEYHBD5PEQEAAAEV'
    'RnYvNxgAPzwvLy/9L/0Q/S88/TwQ/TwBL/2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQB'
    'Li4uLi4uLi4uLi4ALi4uLi4uLi4xMAFJaLkAFQBYSWhhsEBSWDgRN7kAWP/AOFklIyInBiMiJwYjIicGBwYHBiMiJyY1NDc2NzYz'
    'MhUUBwYVFBcWMzI3Njc2JyYnJic3NjMyFxYXFjMyNzY3NjMyFxYzMjc2MzIXFjsBAQcnNxMHJzcPASc3B6VGgTUufHQuP3YsMwQK'
    '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'
    'AwUUbBAHAZhB/MJUexJIEgsuCREWGRIZYhlwNBsLCQEBUyVO/pOdIQQBBwg6IyRDfMCWYIUzLCQ+GSJfFhG0mbz6W3R/abLQXV78'
    '8q0EiUUKORgRMDYRDwgECAAAAwAZ/88D2wYlACcAMQBTAHNALQFUVEBVAlJORDUwLCYhHRcWBQRKQjo4NjUyLiUjGRQRCQIqAQtG'
    'Az4+EQE1RnYvNxgALy8Q/QEv/S4uLi4uLi4uLi4uLi4uLgAuLi4uLi4uLi4uLi4uMTABSWi5ADUAVEloYbBAUlg4ETe5AFT/wDhZ'
    'ARYXBisBFAcGBxYVFAcGBwYnNzY3AicHJjU0NzYzMhcAEzY3Nic3FgM2NTQnBgcWMzIBBwYHNTY3JjU0NzYzMhcWBwYnJiMiBwYV'
    'FBcWMzI3NhcWA9AEBwQSJSMtVywUGCWrwyq7U73wGYUYBgcHIQF0vkgQDhBsI9cBJEBdGBZb/p8QkI8YO1E1OzRFHgkHBxQfNBYa'
    'HyEpPCEgFQQFA/MKFgKgpdR3VUwtND8JKA6NV3IBAvAqkCIjYRke/qv+p3rRtctkdPxICQk2M01HAgTpPh00Lw8hNyYmPkU/EwkJ'
    'CAwNDxMYGyEJBgICAAIAGf+XBKwGJQAwAFIAf0A1AVNTQFQAUU1DNCskHhYVBUlBOTc1NDEqGhELBTAAAgMoAS0DAS0wLwMARQM9'
    'PQkBAAABNEZ2LzcYAD88Ly8Q/RD9PAEv/RD9EP08Li4uLi4uLi4uLi4uAC4uLi4uLi4uLi4xMAFJaLkANABTSWhhsEBSWDgRN7kA'
    'U//AOFklIyIRNDcCBwYHBjU0NzY3NjcmJyYnByYnJicmNzYzMhcWFxYXNjc2JyYnNxYTFjsBAQcGBzU2NyY1NDc2MzIXFgcGJyYj'
    'IgcGFRQXFjMyNzYXFgSsTO8Dk+CNxCp4TEuGdzJ6TqQPNCQ+AQIOAwYGGPx7lzVHLjIDBRRsEAcBmEH8nBCQjxg7UTU7NEUeCQcH'
    'FB80FhofISk8ISAVBAUBAVMlTv6TnSEEAQcIOiMkQ3zAlmCFMywkPhkiXxYRtJm8+lt0f2my0F1e/PKtBLo+HTQvDyE3JiY+RT8T'
    'CQkIDA0PExgbIQkGAgIAAAMAU/35A9sErQAnADEAVgBxQCwBV1dAWAJTTkQwLCEdFxYRBQRVSkI6ODY1LiUjGRQRCQIqAQs+A0Ym'
    'NQEZRnYvNxgALy8v/QEv/S4uLi4uLi4uLi4uLi4uLgAuLi4uLi4uLi4uLi4xMAFJaLkAGQBXSWhhsEBSWDgRN7kAV//AOFkBFhcG'
    'KwEUBwYHFhUUBwYHBic3NjcCJwcmNTQ3NjMyFwATNjc2JzcWAzY1NCcGBxYzMgMHBgc1NjcmNTQ3NjMyFxYHBicmIyIHBhUUFxYz'
    'Mj8BNjMyFxYD0AQHBBIlIy1XLBQYJavDKrtTvfAZhRgGBwchAXS+SBAOEGwj1wEkQF0YFlvDEIyTGTtRNTo1Qx8JBgcUJC8WGh8z'
    'KikgIQsGBAQBAgPzChYCoKXUd1VMLTQ/CSgOjVdyAQLwKpAiI2EZHv6r/qd60bXLZHT8SAkJNjNNRwL+Ij8bNS8PIDcnJT9FQBMJ'
    'CAcNDQ8TJhkWCQMBAQMAAgBM/d8ErASuADAAUgB9QDQBU1NAVABRTUMkHhYVCQVJQTk3NTQxKhoRCwUwAAIDKAEtAwEtMC8DAD0D'
    'RSs0AQAAARpGdi83GAA/PC8vL/0Q/TwBL/0Q/RD9PC4uLi4uLi4uLi4uLgAuLi4uLi4uLi4xMAFJaLkAGgBTSWhhsEBSWDgRN7kA'
    'U//AOFklIyIRNDcCBwYHBjU0NzY3NjcmJyYnByYnJicmNzYzMhcWFxYXNjc2JyYnNxYTFjsBAQcGBzU2NyY1NDc2MzIXFgcGJyYj'
    'IgcGFRQXFjMyNzYXFgSsTO8Dk+CNxCp4TEuGdzJ6TqQPNCQ+AQIOAwYGGPx7lzVHLjIDBRRsEAcBmEH9KRCXiCcsUTU6NUQfCQcH'
    'FCEyFxkfMyopISAVBAUBAVMlTv6TnSEEAQcIOiMkQ3zAlmCFMywkPhkiXxYRtJm8+lt0f2my0F1e/PKt/dg+HzEuGRc3JiU/RT8T'
    'CQgHDAwOFCYZFgkGAgIAAAIAU//PA9sErQAnADEAVkAeATIyQDMCMCwhHRcWBQQuJSMZFBEJAioBCyYRARlGdi83GAAvLwEv/S4u'
    'Li4uLi4uAC4uLi4uLi4uMTABSWi5ABkAMkloYbBAUlg4ETe5ADL/wDhZARYXBisBFAcGBxYVFAcGBwYnNzY3AicHJjU0NzYzMhcA'
    'EzY3Nic3FgM2NTQnBgcWMzID0AQHBBIlIy1XLBQYJavDKrtTvfAZhRgGBwchAXS+SBAOEGwj1wEkQF0YFlsD8woWAqCl1HdVTC00'
    'PwkoDo1XcgEC8CqQIiNhGR7+q/6netG1y2R0/EgJCTYzTUcCAAEATP+XBKwErgAwAGJAJgExMUAyACQeFhUFKhoRCwUwAAIDKAEt'
    'AwEtMC8DACsJAQAAARpGdi83GAA/PC8vEP08AS/9EP0Q/TwuLi4uLgAuLi4uLjEwAUlouQAaADFJaGGwQFJYOBE3uQAx/8A4WSUj'
    'IhE0NwIHBgcGNTQ3Njc2NyYnJicHJicmJyY3NjMyFxYXFhc2NzYnJic3FhMWOwEErEzvA5PgjcQqeExLhncyek6kDzQkPgECDgMG'
    'Bhj8e5c1Ry4yAwUUbBAHAZhBAQFTJU7+k50hBAEHCDojJEN8wJZghTMsJD4ZIl8WEbSZvPpbdH9pstBdXvzyrQAAAQCb/l4FLwJp'
    'ADcAUkAeATg4QDkCHgIgAiQBGiwBEDIBCgYDNigDFjYWARpGdi83GAAvLxD9EP0BL/0v/S/9Li4ALi4xMAFJaLkAGgA4SWhhsEBS'
    'WDgRN7kAOP/AOFkBFgcGJyYjIgcGFRQXFhcWFRQHBgcGIyInJjU0NzY3NhUUBwYVFBcWMzI3NjU0JyYnJjU0NzYzMgUnCAUEHDtb'
    'fmxZssMhB0MslLfspm55dRIREAtLQVmvnK369VshL4KMiZUB1SQFBRAha1k0PiAkOgwbaWtHO0lVXqGzmhcFBBAMFIp3V0ZfJzhl'
    'Ji8SExxDfai1AAEAm/3MBRgAlQAuAFNAHgEvL0AwABMuJxUFABkBDiEBCB0DDC4tDAEAAAEORnYvNxgAPzwvLzwQ/QEv/S/9Li4u'
    'Li4ALjEwAUlouQAOAC9JaGGwQFJYOBE3uQAv/8A4WSUhIgcGFRQEFRQHBiEgETQ2NzY3NhUUBwYVFBcWMzI3NjU0JyYnJjU0NzY3'
    'NjsBBRj+3DYyRQFxV5P+jf5AQC4SDw8MOnxfkpuB6GWhGWUxKS1aprEBERcqClIol0p+AUtLtzwYAQERDxdudoM/MBAdRh0RGwcc'
    'Oi5FOSVJAAQAmwABBYcFIgAJAC8AYABrAKBASAFsbEBtMGJfVUY5NC4lGA8CZl5UQC0dGhQGAgQBACsBCmECVUlIAUVEUlEBVlVc'
    'ATANAylaTk0DMmoDPDsRAx8INzYyAAFARnYvNxgAPzw8Ly/9Lzz9EP08PC/9AS/9Lzz9PC88/TwQ/S/9L/08Li4uLi4uLi4uAC4u'
    'Li4uLi4uLi4uMTABSWi5AEAAbEloYbBAUlg4ETe5AGz/wDhZARQjNjU0JzYzFhcUBiMiJwYjIiY1NDc2MzIVFAYVFDMyNzY3NjMy'
    'FxYzMjU0JzcWARAjIicGKwEiJwYrASInJjU0NzY/AhYdARQXFjsBMjY9ATQnNxEUFxYzMjU0AzcSAScGBwYVFBcWMzID0yoCGxQh'
    'DoUyJyoXJzMeKA0PFAkEISINAQ0FDwkCCiUlBigSAS/HxDc4mxG2IlIOK0FGXD8t5QFXCCYhVA1IVwttLTRjb05mR/x6EEZMc1Y+'
    'MyoEvl0SEUQvKy7yKDsqSzAfFxshDQYZBiAeBDUUDUgnEBUhHvzq/uS4uNViFx41clA5W1hhHFNm7VBGX0nNeYhb/i9oRlJziQGg'
    'Y/6B/vHLFSI0IiIVDwAAAQBVAogBRgN5AAsANkAOAQwMQA0AAAIGCQMBBkZ2LzcYAC8vAS/9ADEwAUlouQAGAAxJaGGwQFJYOBE3'
    'uQAM/8A4WQEUBiMiJjU0NjMyFgFGRzIxR0UxM0gDADJGRzExSEYAAAEAvwNqAQME4AAOADhADwEPD0AQAAgGAQALBAEIRnYvNxgA'
    'Ly8BL/0uADEwAUlouQAIAA9JaGGwQFJYOBE3uQAP/8A4WQEUBwYHJjU0Azc2MxYXFgEDDwkEAyUyBgMCAQYDoRgTCwECDUMBAh4E'
    'AQZLAAEB0wF3AhcC7QAOADhADwEPD0AQAAgGAQALBAEIRnYvNxgALy8BL/0uADEwAUlouQAIAA9JaGGwQFJYOBE3uQAP/8A4WQEU'
    'BwYHJjU0Azc2MxYXFgIXDwkEAyUyBgMCAQYBrhgTCwECDUMBAh4EAQZLAAIAkQMcAjMD4gADAAcAkkBFAQgIQAkABgQCAAIBAgMF'
    'AwABAQIAAAEEBwQFBQUGBwcEBgYHAQABAgUCAwAAAQMDAAcGBwQFBAUGBgcFBQYHAwUBAQZGdi83GAAvPC88AYcuCMQI/AjEhy4I'
    'xAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4AMTABSWi5AAYACEloYbBAUlg4ETe5AAj/wDhZAQcnNw8BJzcCM2NkY3hiZGMDf2Nk'
    'YmNjZGIAAgGZ/LkDO/1/AAMABwCSQEUBCAhACQAGBAIAAgECAwUDAAEBAgAAAQQHBAUFBQYHBwQGBgcBAAECBQIDAAABAwMABwYH'
    'BAUEBQYGBwUFBgcDBQEBBkZ2LzcYAC88LzwBhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLgAxMAFJaLkABgAI'
    'SWhhsEBSWDgRN7kACP/AOFkBByc3DwEnNwM7Y2RjeGJkY/0cY2RiY2NkYgAAAAAAGAEmAAAAAAAAAAAALAD8AAAAAAAAAAEAEgAA'
    'AAAAAAAAAAIADgEoAAAAAAAAAAMAGgE2AAAAAAAAAAQAEgAAAAAAAAAAAAUAMgASAAAAAAAAAAYAEABEAAAAAAAAAAcAOABUAAEA'
    'AAAAAAAAFgCMAAEAAAAAAAEACQCiAAEAAAAAAAIABwCrAAEAAAAAAAMADQCyAAEAAAAAAAQACQCiAAEAAAAAAAUAGQC/AAEAAAAA'
    'AAYACADYAAEAAAAAAAcAHADgAAMAAQQJAAAALAD8AAMAAQQJAAEAEgAAAAMAAQQJAAIADgEoAAMAAQQJAAMAGgE2AAMAAQQJAAQA'
    'EgAAAAMAAQQJAAUAMgASAAMAAQQJAAYAEABEAAMAAQQJAAcAOABUAEIAIABOAGEAegBhAG4AaQBuAFYAZQByAHMAaQBvAG4AIAAy'
    'AC4AMAAxACAALQAgAEIAdQBpAGwAZAAgADEAMwA3ADkAQgBOAGEAegBhAG4AaQBuAFAAYQByAHMAYQAgADIAMAAwADEArgAgAC0A'
    'IABCAG8AcgBuAGEAIABSAGEAeQBhAG4AZQBoAK4oYykgMjAwMCBCb3JuYSBSYXlhbmVoQiBOYXphbmluUmVndWxhckJvcm5hIE5h'
    'emFuaW5WZXJzaW9uIDIuMDEgLSBCdWlsZCAxMzc5Qk5hemFuaW5QYXJzYSAyMDAxqCAtIEJvcm5hIFJheWFuZWioACgAYwApACAA'
    'MgAwADAAMAAgAEIAbwByAG4AYQAgAFIAYQB5AGEAbgBlAGgAUgBlAGcAdQBsAGEAcgBCAG8AcgBuAGEAIABOAGEAegBhAG4AaQBu'
    'AAAAAgAAAAAAAP1YADwAAAAAAAAAAAAAAAAAAAAAAAAAAADdAAAAAQACAAMABAAIAAsADAANAA4ADwAQABEAEgATABQAFQAWABcA'
    'GAAZABoAGwAcAB0AIAA+AEAAXgBgAKkAqgDwALgBAgEDAQQBBQC2ALcAtAC1AL4AvwEGAQcBCAEJAQoBCwEMAQ0BDgEPARABEQES'
    'ARMBFAEVARYBFwEYARkBGgEbARwBHQEeAR8BIAEhASIBIwEkASUBJgEnASgBKQEqASsBLAEtAS4BLwEwATEBMgEzATQBNQE2ATcB'
    'OAE5AToBOwE8AT0BPgE/AUABQQFCAUMBRAFFAUYBRwFIAUkBSgFLAUwBTQFOAU8BUAFRAVIBUwFUAVUBVgFXAVgBWQFaAVsBXAFd'
    'AV4BXwFgAWEBYgFjAWQBZQFmAWcBaAFpAWoBawFsAW0BbgFvAXABcQFyAXMBdAF1AXYBdwF4AXkBegF7AXwBfQF+AX8BgAGBAYIB'
    'gwGEAYUBhgGHAYgBiQGKAYsBjAGNAY4BjwGQAZEBkgGTAZQBlQGWAZcBmAGZAZoBmwGcAZ0BngGfAaABoQGiAaMBpAGlAaYBpwGo'
    'AakBqgGrAawBrQGuAa8BsAGxAMMAAAAAAAAAAAp6ZXJvbm9qb2luCHplcm9qb2luC2xlZnR0b3JpZ2h0C3JpZ2h0dG9sZWZ0BXUw'
    'NjBDBXUwNjFCBXUwNjFGBXUwNjQwBXUwNjRCBXUwNjRDBXUwNjREBXUwNjRFBXUwNjRGBXUwNjUwBXUwNjUxBXUwNjUyBXVFODIw'
    'BXVFODIxBXVFODIyBXVFODIzBXVFODI0BXVFODI1BXVFODI2BXVFODI3BXVFODI4BXVFODI5BXVFODJBBXVFODJCBXVFODJDBXVF'
    'ODJEBXVGQzYwBXVGQzYxBXVFODE4BXVGQzVFBXVGQzYyBXVGQzVGBXUwNjIxBXUwNjIyC0hjaXJjdW1mbGV4BXUwNjIzBXVGRTg0'
    'BXUwNjI0BXVGRTg2BXUwNjI1BXVGRTg4BXUwNjI2BXVGRThBBXVGRThCBXVGRThDBXUwNjI3BXVGRThFBXUwNjI4BXVGRTkwBXVG'
    'RTkxBXVGRTkyBXUwNjdFBXVGQjU3BXVGQjU4BXVGQjU5BXUwNjJBBXVGRTk2BXVGRTk3BXVGRTk4BXUwNjJCBXVGRTlBBXVGRTlC'
    'BXVGRTlDBXUwNjJDBXVGRTlFBXVGRTlGBXVGRUEwBXUwNjg2BXVGQjdCBXVGQjdDBXVGQjdEBXUwNjJEBXVGRUEyBXVGRUEzBXVG'
    'RUE0BXUwNjJFBXVGRUE2BXVGRUE3BXVGRUE4BXUwNjJGBXVGRUFBBXUwNjMwBXVGRUFDBXUwNjMxBXVGRUFFBXUwNjMyBXVGRUIw'
    'BXUwNjk4BXVGQjhCBXUwNjMzBXVGRUIyBXVGRUIzBXVGRUI0BXUwNjM0BXVGRUI2BXVGRUI3BXVGRUI4BXUwNjM1BXVGRUJBBXVG'
    'RUJCBXVGRUJDBXUwNjM2BXVGRUJFBXVGRUJGBXVGRUMwBXUwNjM3BXVGRUMyBXVGRUMzBXVGRUM0BXUwNjM4BXVGRUM2BXVGRUM3'
    'BXVGRUM4BXUwNjM5BXVGRUNBBXVGRUNCBXVGRUNDBXUwNjNBBXVGRUNFBXVGRUNGBXVGRUQwBXUwNjQxBXVGRUQyBXVGRUQzBXVG'
    'RUQ0BXUwNjQyBXVGRUQ2BXVGRUQ3BXVGRUQ4BXUwNjQzBXVGRURBBXVGRURCBXVGRURDBXUwNkFGBXVGQjkzBXVGQjk0BXVGQjk1'
    'BXUwNjQ0BXVGRURFBXVGRURGBXVGRUUwBXUwNjQ1BXVGRUUyBXVGRUUzBXVGRUU0BXUwNjQ2BXVGRUU2BXVGRUU3BXVGRUU4BXUw'
    'NjQ4BXVGRUVFBXUwNjQ3BXVGRUVBBXVGRUVCBXVGRUVDBXUwNjI5BXVGRTk0BXUwNjRBBXVGRUYyBXVGRUYzBXVGRUY0BXVGRUY1'
    'BXVGRUY2BXVGRUY3BXVGRUY4BXVGRUY5BXVGRUZBBXVGRUZCBXVGRUZDBXUwNjQ5BXVGRUYwBXVGREYyAAAAAQAAgAAAAAH0BAAA'
    'AGAABT8CdUIgTmF6YW5pbiAgICAgICD/////N////kIgTlIwMAAAAAAAAAABAAAACgAmAGgAAWFyYWIACAAEAAAAAP//AAUAAAAB'
    'AAIAAwAEAAVpbml0ACBtZWRpACZmaW5hACxsaWdhADJtc2V0ADoAAAABAAAAAAABAAEAAAABAAIAAAACAAMABAAAAAIABQAHAAkA'
    'FACSAQwBygIyApgERgSMBnYAAQABAAEACAACADwAGwBXAF0AYQBlAGkAbQBxAHUAeQB7AIcAiwCPAJMAlwCbAJ8AowCnAKsArwCz'
    'ALcAuwC/AMUAywABABsAVQBbAF8AYwBnAGsAbwBzAHcAewCFAIkAjQCRAJUAmQCdAKEApQCpAK0AsQC1ALkAvQDDAMkAAQABAAEA'
    'CAACADoAGgBYAF4AYgBmAGoAbgByAHYAegCIAIwAkACUAJgAnACgAKQAqACsALAAtAC4ALwAwADGAMwAAQAaAFUAWwBfAGMAZwBr'
    'AG8AcwB3AIUAiQCNAJEAlQCZAJ0AoQClAKkArQCxALUAuQC9AMMAyQABAAEAAQAIAAIAXAArAE4AUABSAFQAVgBaAFwAYABkAGgA'
    'bABwAHQAeAB8AH4AgACCAIQAhgCKAI4AkgCWAJoAngCiAKYAqgCuALIAtgC6AL4AwgDEAMgAygDUAM4A0ADSANYAAQArAE0ATwBR'
    'AFMAVQBZAFsAXwBjAGcAawBvAHMAdwB7AH0AfwCBAIMAhQCJAI0AkQCVAJkAnQChAKUAqQCtALEAtQC5AL0AwQDDAMcAyQDTAM0A'
    'zwDRANUABAAJAAEACAABAFgAAgAKADYABQAMABIAGAAeACQAzQACAE4AzwACAFAA0QACAFQA0wACAFoA1wADALgAxAAEAAoAEAAW'
    'ABwAzgACAE4A0AACAFAA0gACAFQA1AACAFoAAQACALcAuAAEAAcAAQAIAAEAVgACAAoAPAAGAA4AFAAaACAAJgAsAEgAAgAwAEkA'
    'AgAxAEYAAgAzAEcAAgA0AEoAAgA1AEsAAgAyAAMACAAOABQA2wACADAA2QACADMA3AACADIAAQACADYANwAFAAEAAQAIAAIAHAE2'
    'AAMAAAAAAA4AAQAEAAIAAQABAAEABgABAIsATABNAE4ATwBQAFEAUgBTAFQAVQBWAFcAWABZAFoAWwBcAF0AXgBfAGAAYQBiAGMA'
    'ZABlAGYAZwBoAGkAagBrAGwAbQBuAG8AcABxAHIAcwB0AHUAdgB3AHgAeQB6AHsAfAB9AH4AfwCAAIEAggCDAIQAhQCGAIcAiACJ'
    'AIoAiwCMAI0AjgCPAJAAkQCSAJMAlACVAJYAlwCYAJkAmgCbAJwAnQCeAJ8AoAChAKIAowCkAKUApgCnAKgAqQCqAKsArACtAK4A'
    'rwCwALEAsgCzALQAtQC2ALcAuAC5ALoAuwC8AL0AvgC/AMAAwQDCAMMAxADFAMYAxwDIAMkAygDLAMwAzQDOAM8A0ADRANIA0wDU'
    'ANUA1gACABIATABMAAIATQBaAAEAVwBkAAIAawB2AAIAewB8AAIAfwB/AAIAgQCBAAIAhQCKAAIAjQCSAAIAnQCgAAIAogCiAAEA'
    'pQCmAAIAsQCyAAIArQCuAAIAuQC8AAIAwQDGAAIAyQDMAAIA1QDWAAIAAQABAAEACAACACAADQA7ADwAOAA5AD0AOgBCAEMARQBA'
    'AEEARADaAAEADQAwADEAMwA0ADYANwBIAEkASwBGAEcASgDZAAUAAQABAAgAAgAcATYAAwAAAAAADgABAAQAAgABAAEAAQAIAAEA'
    'iwBMAE0ATgBPAFAAUQBSAFMAVABVAFYAVwBYAFkAWgBbAFwAXQBeAF8AYABhAGIAYwBkAGUAZgBnAGgAaQBqAGsAbABtAG4AbwBw'
    'AHEAcgBzAHQAdQB2AHcAeAB5AHoAewB8AH0AfgB/AIAAgQCCAIMAhACFAIYAhwCIAIkAigCLAIwAjQCOAI8AkACRAJIAkwCUAJUA'
    'lgCXAJgAmQCaAJsAnACdAJ4AnwCgAKEAogCjAKQApQCmAKcAqACpAKoAqwCsAK0ArgCvALAAsQCyALMAtAC1ALYAtwC4ALkAugC7'
    'ALwAvQC+AL8AwADBAMIAwwDEAMUAxgDHAMgAyQDKAMsAzADNAM4AzwDQANEA0gDTANQA1QDWAAIAHABMAFAAAgBRAFYAAQBXAFoA'
    'AgBbAGIAAQBjAGoAAgBrAG4AAQBvAHQAAQB1AHYAAgB3AHgAAQB5AH4AAgB/AIYAAQCHAIgAAgCJAIoAAQCLAIwAAgCNAI4AAQCP'
    'AJAAAgCRAJIAAQCTAJQAAgCVAJwAAgCfAKAAAgCjAKgAAgCrALQAAgC3ALgAAgC7ALwAAgC/AMAAAgDDAMUAAgDNANAAAgDTANQA'
    'AgABAAEAAQAIAAIACgACAD8APgABAAIAMgA1'
)

BCD_BNAZANIN_BOLD_TTF_BASE64 = (
    'AAEAAAAQAQAABAAAR1NVQiToNn4AAAEMAAAG1k9TLzK4ReMDAAAH5AAAAFZQQ0xUnky0vgAACDwAAAA2Y21hcBsEsYcAAAh0AAAD'
    '7mN2dCDPn8xyAAAMZAAAAHRmcGdtgzPCTwAADNgAAAAUZ2x5Zkem8XMAAAzsAAC/KGhkbXjjb+zIAADMFAAADchoZWFk00MIOgAA'
    '2dwAAAA2aGhlYQy2A6YAANoUAAAAJGhtdHjvj0vIAADaOAAAA2Rsb2NhAEYoQgAA3ZwAAANobWF4cAGGAhwAAOEEAAAAIG5hbWUk'
    'CeWBAADhJAAAArJwb3N0dsRTtwAA49gAAAYOcHJlcOc9+DcAAOnoAAAAcAABAAAACgAmAGgAAWFyYWIACAAEAAAAAP//AAUAAAAB'
    'AAIAAwAEAAVpbml0ACBtZWRpACZmaW5hACxsaWdhADJtc2V0ADoAAAABAAAAAAABAAEAAAABAAIAAAACAAMABAAAAAIABQAHAAkA'
    'FACSAQwBygIyAnoEKARqBlQAAQABAAEACAACADwAGwBXAF0AYQBlAGkAbQBxAHUAeQB7AIcAiwCPAJMAlwCbAJ8AowCnAKsArwCz'
    'ALcAuwC/AMUAywABABsAVQBbAF8AYwBnAGsAbwBzAHcAewCFAIkAjQCRAJUAmQCdAKEApQCpAK0AsQC1ALkAvQDDAMkAAQABAAEA'
    'CAACADoAGgBYAF4AYgBmAGoAbgByAHYAegCIAIwAkACUAJgAnACgAKQAqACsALAAtAC4ALwAwADGAMwAAQAaAFUAWwBfAGMAZwBr'
    'AG8AcwB3AIUAiQCNAJEAlQCZAJ0AoQClAKkArQCxALUAuQC9AMMAyQABAAEAAQAIAAIAXAArAE4AUABSAFQAVgBaAFwAYABkAGgA'
    'bABwAHQAeAB8AH4AgACCAIQAhgCKAI4AkgCWAJoAngCiAKYAqgCuALIAtgC6AL4AwgDEAMgAygDUAM4A0ADSANYAAQArAE0ATwBR'
    'AFMAVQBZAFsAXwBjAGcAawBvAHMAdwB7AH0AfwCBAIMAhQCJAI0AkQCVAJkAnQChAKUAqQCtALEAtQC5AL0AwQDDAMcAyQDTAM0A'
    'zwDRANUABAAJAAEACAABAFgAAgAKADYABQAMABIAGAAeACQAzQACAE4AzwACAFAA0QACAFQA0wACAFoA1wADALgAxAAEAAoAEAAW'
    'ABwAzgACAE4A0AACAFAA0gACAFQA1AACAFoAAQACALcAuAAEAAcAAQAIAAEAOgABAAgABgAOABQAGgAgACYALABIAAIAMABJAAIA'
    'MQBGAAIAMwBHAAIANABKAAIANQBLAAIAMgABAAEANgAFAAEAAQAIAAIAHAE2AAMAAAAAAA4AAQAEAAIAAQABAAEABgABAIsATABN'
    'AE4ATwBQAFEAUgBTAFQAVQBWAFcAWABZAFoAWwBcAF0AXgBfAGAAYQBiAGMAZABlAGYAZwBoAGkAagBrAGwAbQBuAG8AcABxAHIA'
    'cwB0AHUAdgB3AHgAeQB6AHsAfAB9AH4AfwCAAIEAggCDAIQAhQCGAIcAiACJAIoAiwCMAI0AjgCPAJAAkQCSAJMAlACVAJYAlwCY'
    'AJkAmgCbAJwAnQCeAJ8AoAChAKIAowCkAKUApgCnAKgAqQCqAKsArACtAK4ArwCwALEAsgCzALQAtQC2ALcAuAC5ALoAuwC8AL0A'
    'vgC/AMAAwQDCAMMAxADFAMYAxwDIAMkAygDLAMwAzQDOAM8A0ADRANIA0wDUANUA1gACABIATABMAAIATQBaAAEAVwBkAAIAawB2'
    'AAIAewB8AAIAfwB/AAIAgQCBAAIAhQCKAAIAjQCSAAIAnQCgAAIAogCiAAEApQCmAAIAsQCyAAIArQCuAAIAuQC8AAIAwQDGAAIA'
    'yQDMAAIA1QDWAAIAAQABAAEACAACAB4ADAA7ADwAOAA5AD0AOgBCAEMARQBAAEEARAABAAwAMAAxADMANAA2ADcASABJAEsARgBH'
    'AEoABQABAAEACAACABwBNgADAAAAAAAOAAEABAACAAEAAQABAAgAAQCLAEwATQBOAE8AUABRAFIAUwBUAFUAVgBXAFgAWQBaAFsA'
    'XABdAF4AXwBgAGEAYgBjAGQAZQBmAGcAaABpAGoAawBsAG0AbgBvAHAAcQByAHMAdAB1AHYAdwB4AHkAegB7AHwAfQB+AH8AgACB'
    'AIIAgwCEAIUAhgCHAIgAiQCKAIsAjACNAI4AjwCQAJEAkgCTAJQAlQCWAJcAmACZAJoAmwCcAJ0AngCfAKAAoQCiAKMApAClAKYA'
    'pwCoAKkAqgCrAKwArQCuAK8AsACxALIAswC0ALUAtgC3ALgAuQC6ALsAvAC9AL4AvwDAAMEAwgDDAMQAxQDGAMcAyADJAMoAywDM'
    'AM0AzgDPANAA0QDSANMA1ADVANYAAgAcAEwAUAACAFEAVgABAFcAWgACAFsAYgABAGMAagACAGsAbgABAG8AdAABAHUAdgACAHcA'
    'eAABAHkAfgACAH8AhgABAIcAiAACAIkAigABAIsAjAACAI0AjgABAI8AkAACAJEAkgABAJMAlAACAJUAnAACAJ8AoAACAKMAqAAC'
    'AKsAtAACALcAuAACALsAvAACAL8AwAACAMMAxQACAM0A0AACANMA1AACAAEAAQABAAgAAgAKAAIAPwA+AAEAAgAyADUAAAABAAAC'
    'vAAFAAAFmgUzAAABJQWaBTMAAAOgAGYCEgAAAAAHAAAAAAAAAAAAYACAAAAAAAAACAAAAABNWjczACAAIP78CE77oAAACE4EYAAA'
    'AEAAAAAAAAAAAQAAgAAAAAG4BAAAAGAABT0CdUIgTmF6YW5pbiAgIEJkICD/////N////kIgTkIwMAMAAAAAAAAAAAIAAQAAAAAA'
    'FAADAAEAAAEaAAABBgAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADBAAAAAUAAAYHCAkKCwwNDg8QERITFBUWFxgA'
    'ABkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABoAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAdAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHgAAAAAAAAAAAAAA2AAAAB8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEAAAAAAAAAAAAEAtQAAABaAEAABQAaACEAJQA6AD0AWwBdAHsA'
    'fQCrALcAuwDXAPcC2QYMBhsGHwY6BlIGaQZ+BoYGmAapBq8GwAbMBvkgDyAZIB0gOiIZ6BjoLftZ+337i/uV+6X7//xi/fL+/P//'
    'AAAAIAAlACgAPQBbAF0AewB9AKsAtwC7ANcA9wLZBgwGGwYfBiEGQAZgBn4GhgaYBqkGrwbABswG8CAMIBggHCA5IhnoGOgg+1b7'
    'evuK+477pPv8/F798v6B////4//g/97/3P+//77/of+g/3MAIf9k/0n/Kv3/+iD6EvoPAAAAAPmu+eH56fnr+gT6AvoH+f35HuAW'
    '4A7gDN/x3r8YMBgYBQkE9QT5BR8FIwTNAAAC5QAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAagAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGIAAABoAAAATABNAE8AUQBTAFUAWQBbAMcAYwBnAGsAcwB3'
    'AHsAfQB/AIEAhQCJAI0AkQCVAJkAnQChAC8ApQCpAK0AtQC5AL0AwwDBANUAyQAwADEAMgAzADQANQA2ADcASQBLAEYARwBKAE0A'
    'TgBPAFAAUQBSAFMAVABVAFYAVwBYAFkAWgBbAFwAXQBeAMcAyABjAGQAZQBmAGcAaABpAGoAawBsAG0AbgBzAHQAdQB2AHcAeAB5'
    'AHoAewB8AH0AfgB/AIAAgQCCAIUAhgCHAIgAiQCKAIsAjACNAI4AjwCQAJEAkgCTAJQAlQCWAJcAmACZAJoAmwCcAJ0AngCfAKAA'
    'oQCiAKMApAClAKYApwCoAKkAqgCrAKwArQCuAK8AsAC1ALYAtwC4ALkAugC7ALwAvQC+AL8AwADDAMQAxQDGAMEAwgDVANYAyQDK'
    'AMsAzADNAM4AzwDQANEA0gDTANQAAAAAAFwAmwDlAR4A2ACGAHMBdQLKAyMCnQOfAMoDdgLWA9gClwMtA55fsl+yX7Jfsl+yX7Jf'
    'sl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7Jfsl+yX7IAJAAmQAEALHZFILADJUUj'
    'YWgYI2hgRC0AAgCAAAADgAU9AAMABwBWQCABCAhACQIHBAEBAAYFAQMCBQQDAAcGAwECAQMAAAEARnYvNxgAPzwvPBD9PBD9PAEv'
    'PP08Lzz9PAAxMAFJaLkAAAAISWhhsEBSWDgRN7kACP/AOFkzESERJSERIYADAP2AAgD+AAU9+sOABD0AAgC+AAEBjwTBAA8AGwBM'
    'QBsBHBxAHRAGBRYNBhAAAgwGAQUZAxMOEwABDEZ2LzcYAD8vEP0BL/0v/TwQ/QAuLjEwAUlouQAMABxJaGGwQFJYOBE3uQAc/8A4'
    'WQEUBwYPASMmJyYnJjU0MzITFAYjIiY1NDYzMhYBjRcbBQ8+Aw4GHBhnaAI7Kys5OiopPQPOR5GnVOYnvVKulULx+6UrOjorKjo7'
    'AAMAe/+bA5YE7QALAA8AGwBZQCABHBxAHRAODA4PBw0MDA0AAgYWAhADAwkZAxMPDQEGRnYvNxgALy8v/S/9AS/9L/2HLg7EDvwO'
    'xAEuLgAxMAFJaLkABgAcSWhhsEBSWDgRN7kAHP/AOFkBFAYjIiY1NDYzMhYlAScBExQGIyImNTQ2MzIWAUk7LCs8PSorPAHs/iJi'
    'AeHAOystOjwrKjwEESw6OysqPDuv+rABBVH7ays8Oi0rOzwAAAEA8v8gApUEsQAWADlADwEXF0AYDg4LBwARBAEARnYvNxgALy8B'
    'Li4uLgAxMAFJaLkAAAAXSWhhsEBSWDgRN7kAF//AOFkTEAEWMzI2NRQnAhE0Ezc0JiMiBwYHBvIBClURCR9BusJEHQgUVm1HYAHa'
    '/rf+6FkQCAVsATQBCf0BTnUGD1pyi70AAAEAtf8gAlgEsQAYADpAEAEZGUAaABAJDQIAEwYBEEZ2LzcYAC8vAS/9Li4AMTABSWi5'
    'ABAAGUloYbBAUlg4ETe5ABn/wDhZARQHBgcGIyImNRQ3EhE0Ayc0NjMyFxYXFgJYXERpVRIJH0G6wkQdCBRWbUdgAdq4tYZuWRAI'
    'BWwBNAEJ/QFOdQYPWnKLvQAJALAABgRPA6QAEQAeACsANwBGAFkAZgBzAIQAAAAHBgcGIyInJjc2NzYzMhYVFCUUBwYjIicmNTQz'
    'MhYBFAYjIicmJzY3NjMyBRQGIyImNTQ2MzIWFxYVFAYjIicmJyY1NDMWJSInJicmNTQ2MzIXFhcWFxYVFAcUBwYjIjU0NjMyFxYT'
    'FCMiJjU0NzYzMhcWJwYHBiMiJjU0NzY3NjMyFxYDuTRPSAUEBAIDAyhKGSEjMf72TQYEBQZIViA0AXowHnV5CQEBCX5sUv6GMiMj'
    'MjEkIzLzGTQjIxRPIgIHlP6dBQSPShgzIyYaDB0wEAIpCntvUjAecH4K3lYfNU0GBAUGSLQnTBkhIjJbUUYFBAQCAgKXHy8QAgIC'
    'C5JHGDMjI5lwfgoKf2tRLv5dHjRMBgQGBUhVIzAwIyQxMvQaISMxFFCOBQQIJeECJkwZISMxHQ0xUUYEBQdhBQZGVR81TgX+gVMw'
    'H29+CgqAoo1KGTQiNTUwDwICBAABAHL/+gOJAxEACwBlQCsBDAxADQALAA0BBgUNAwoJAgMBAQgHBAMDCwoHAwYDBQQBAwAJCAMC'
    'AQVGdi83GAAvPC88Lxc8/Rc8AS8XPP0XPBD9PBD9PAAxMAFJaLkABQAMSWhhsEBSWDgRN7kADP/AOFkBIREjESE1IREzESEDif6n'
    'Zf6nAVllAVkBUv6oAVhmAVn+pwABALb/NwHhAPIADwBBQBQBEBBAEQYLCwkGAgADCQgNAAEARnYvNxgAPy88LwEv/S4uAC4xMAFJ'
    'aLkAAAAQSWhhsEBSWDgRN7kAEP/AOFk3NDYzMhYVFAcjNjUGIyImtk44RGGeZ4IZGTNDcTdKZ0SQgHBfCUEAAQByAVIDiQG4AAMA'
    'PUARAQQEQAUAAwIBAAMCAQABAUZ2LzcYAC88LzwBLi4uLgAxMAFJaLkAAQAESWhhsEBSWDgRN7kABP/AOFkBITUhA4n86QMXAVJm'
    'AAEAsv//AZgA5gALADdADwEMDEANAAACBgkDAAEGRnYvNxgAPy8BL/0AMTABSWi5AAYADEloYbBAUlg4ETe5AAz/wDhZJRQGIyIm'
    'NTQ2MzIWAZhCMTFCQzAwQ3MxQ0MxMENDAAABAIb/lQIRBOUAAwBPQBwBBARABQACAAADAAEHAQIDAwACAgMDAAIBAQJGdi83GAAv'
    'PC88AYcuCMQI/AjEAS4uADEwAUlouQACAARJaGGwQFJYOBE3uQAE/8A4WQkBIwECEf7gawEfBOX6sAVQAAEAewDoAmIC1gAPADVA'
    'DQEQEEARAAgADAQBCEZ2LzcYAC8vAS4uADEwAUlouQAIABBJaGGwQFJYOBE3uQAQ/8A4WQEGBwYHJicmJzY3NjcWFxYCYjZJJUs6'
    'Ox9kPj8jUDswJQHfMUckW0ZAIU00PiJmRjQoAAABAIz/qwIsBMwADwA4QA8BEBBAEQAKCAIADAQBCkZ2LzcYAC8vAS/9LgAxMAFJ'
    'aLkACgAQSWhhsEBSWDgRN7kAEP/AOFkBFAcGIyInJjUQATY3FhcSAiweITMgCAP+/R2BO1F2AWyphZNCGV0CAgFyMMUowf7oAAAB'
    'AGz/rwN1BNYAHgBFQBYBHx9AIAAdEgYACAIOBgQEFxMKARJGdi83GAAvLy/9PAEv/S4uLgAuMTABSWi5ABIAH0loYbBAUlg4ETe5'
    'AB//wDhZARQHBiMiJxYRECMiJyYnJicmAzcWFxYzMjc2NzYzMgN1WWGlEhE/YSMKAgMLLUC7kRtHTYBlPSMiGxUyBJepcXoC7/7k'
    '/rVlP0CNqvIBI/cgUlQzHUAzAAEAZf+uBA8E2QAxAElAGAEyMkAzADAkHgcXCQALAhMEAyoYDwEXRnYvNxgALy8v/QEv/S4uLgAu'
    'Li4uMTABSWi5ABcAMkloYbBAUlg4ETe5ADL/wDhZARQHBiMiJicGBxYRFAcGIyInJicmJwInExYXFhcWMzI3Njc2MzIXFhcWMzI3'
    'Njc2MzIEDzA9aTg3I0J1QxQaMCcKAwILM1KqkxkmKRcuOk0hDhEMHigKEQIYSTkeCxcQFzMEeXNsihkrYwXH/uKBXnphQECPuwEu'
    'zwEDFEFFFi03F043MlcEMjMSSzMAAAEAa/+cA5sEvQAuAFFAHAEvL0AwAC0eFhQGJRwSCAYABAMpIQMYGAwBEkZ2LzcYAC8vEP0v'
    '/QEuLi4uLi4ALi4uLi4xMAFJaLkAEgAvSWhhsEBSWDgRN7kAL//AOFkBFAcGIyInFhUUBwYjIicmJwInNjcWFzYzMhcWFRQjIiYj'
    'IgcGFRQXFjMyNzYzMgObYW+aIiMmDRc5MQgQFD6vLmY2U2iyS0A1GhdyH1MuJAQiiFxIPwUKA4dfYnEG1jWCbsSH+mcBO+NXriyG'
    'yDwyGykiNSozDAtdPDUAAgB5/68D3wTVABkAMQBPQBwBMjJAMwAqHgYREBoBACICDDADBCQECBQIAQxGdi83GAAvLxD9L/0BL/0v'
    '/S4uAC4uLjEwAUlouQAMADJJaGGwQFJYOBE3uQAy/8A4WQEUBwYjIicGIyInJjU0EzY3Jzc2MzIXEhcWBzQnJicGBwYVFDMyNzY3'
    'NjMyFxYXFjMyA987RodtOE9+ckA6clhXJ2gJCQcO7WqGhJN8ZjtFTG5IHQsSDCAoDBQFHE9tAR6PX3FqelFKdYABB8uQLfEWEv7U'
    'wfXMVdOyZ2u0xkN3LRBELStICCsAAAEAcP+5A4UEzAAvAElAGAEwMEAxAC4rIB4UDgAnAhYjAxoaDAEORnYvNxgALy8Q/QEv/S4u'
    'Li4ALi4uMTABSWi5AA4AMEloYbBAUlg4ETe5ADD/wDhZAQYHBgcGBwYDBgcGIyI1NDc2NzY3JjU0NzYzMhcWFRQjIiYjIgcGFRQX'
    'FjMyJDMyA34GGx4NFzyr0VJKJRoYAg1cKnbedKN4R0dKFBeWKGRDOFNGSzABEhESA04pZnUKCBZC/vVoeDw0BhSBtFKSNstliL46'
    'PUUmMDwyM0YtJpwAAAEAbv++A/sE2AAdADxAEQEeHkAfAAwAFQQGGw8GAQxGdi83GAAvLzwQ/QEuLgAxMAFJaLkADAAeSWhhsEBS'
    'WDgRN7kAHv/AOFkBBgMGBwYjIicmJwInEzYzMhcWFxYXNjcSNzYzMhcD+7lgJDclIikdODJrt1wJBwYRbWJTKSFIXHMOBwYKA63n'
    '/vpi+Kh87IsBJswBGRwTevrUy8PJAQGIERwAAQCQ/7sEHQTWAB0AOUAPAR4eQB8MFQwABhsPAQBGdi83GAAvPC8BLi4ALjEwAUlo'
    'uQAAAB5JaGGwQFJYOBE3uQAe/8A4WTc2EzY3NjMyFxYXEhcDBiMiJyYnJicGBwYHBiMiJ5C5YCQ3JSIpHTY0bbVcCQcGEm1iUikh'
    'SVxzDgYHCufmAQZj+Kh96I7+18r+5xwUevrTy8LJ/4oRGwACAHX/qwNyBNEAEgAdAEhAGAEeHkAfAAUaEwAPAgkHBBwWBA0NAQEJ'
    'RnYvNxgALy8Q/S/9AS/9Li4uAC4xMAFJaLkACQAeSWhhsEBSWDgRN7kAHv/AOFklAyYnJjcGIyA1NDc2MyAREBcWAS4BIyIHBhUU'
    'MzIDclKpMhYBUzP+y0JbkwElOR3+9AFePzUfF3h3xP7nfdthwgjOaKDd/iX+74hFAhg/WiIaHEwAAgCy//8BmAL/AAsAFwBFQBcB'
    'GBhAGQAMAAISBgkDAw8DFRUDAAEGRnYvNxgAPy8Q/RD9AS88/TwAMTABSWi5AAYAGEloYbBAUlg4ETe5ABj/wDhZJRQGIyImNTQ2'
    'MzIWERQGIyImNTQ2MzIWAZhCMTFCQzAwQ0MwMUJDMDBDczFDQzEwQ0MB6TBEQzEwQ0MAAgB2AOADSAI0AAMABwBTQB0BCAhACQAH'
    'BgUEAwIBAAMCAwAFBAMGBwYBAAEBRnYvNxgALzwvPBD9PBD9PAEuLi4uLi4uLgAxMAFJaLkAAQAISWhhsEBSWDgRN7kACP/AOFkl'
    'ITUhNSE1IQNI/S4C0v0uAtLgZohmAAABANb/mwIbBggABwBXQCEBCAhACQAHBAMDAAIBBgUCAgEHBgMABQQDAgMCAQABAUZ2LzcY'
    'AC88LzwQ/TwQ/TwBLzz9PBD9FzwAMTABSWi5AAEACEloYbBAUlg4ETe5AAj/wDhZBSERIRUjETMCG/67AUW2tmUGbT76DwAAAQCH'
    '/5sBowYGAAcAV0AhAQgIQAkABgUCAwECAAQDAgcAAwIDAAUEAwYHBgEAAQFGdi83GAAvPC88EP08EP08AS88/TwQ/Rc8ADEwAUlo'
    'uQABAAhJaGGwQFJYOBE3uQAI/8A4WQUhNTMRIzUhAaP+5I2NARxlPgXwPQABAR/+wANABgYAKgBWQCABKytALAAqGBcAIgENDCgB'
    'BCQBCAgPASAcARMXAAEMRnYvNxgALy8BL/0v/TwQ/S/9Lzz9Li4uLgAxMAFJaLkADAArSWhhsEBSWDgRN7kAK//AOFkBJicmNTQ3'
    'NjU0JyYnNTY1NCcmNTQ3NjcVBgcGFRQXFhUUBQQVFAcGFRQXA0CYYWcaE0hDY+8QG2VgmF49Qw0e/uIBGxsR4v7AG2Zrlj10VRth'
    'Qj4LJjSnL0t+PpdqZhkoDjtAWy5FoDLgcHTqMYJSNbQ1AAABAJb+wAK3BgYAKgBWQCABKytALAAeHQsKDwEGEwECAiYBFxsBIhUB'
    'KgAeCgEKRnYvNxgALy8BLzz9L/0v/TwQ/S/9Li4uLgAxMAFJaLkACgArSWhhsEBSWDgRN7kAK//AOFkBBhUUFxYVFAcGBzU2NzY1'
    'NCcmNTQlJDU0NzY1NCc1FhcWFRQHBhUUFxYXArfvDxxlYJhePUIMHgEd/uYaEuKYYWcbEklAZQJRNKg0RoM4l2pmGSgOOz9cMEKk'
    'L99xdOk3fVcwtDUkHGVqlzl4UCFhQToQAAACAEYAgQMkA5oABQALAE1AGwEMDEANAAsKCAYFBAIACgkEAwMHBgEDAAEIRnYvNxgA'
    'Lxc8Lxc8AS4uLi4uLi4uADEwAUlouQAIAAxJaGGwQFJYOBE3uQAM/8A4WSUjCQEzCwEjCQEzAwMkXf6sAVRX4ktZ/q4BVFfigQGQ'
    'AYn+d/5wAZABif53AAACADIAgQMQA5oABQALAE1AGwEMDEANAAoJCAYEAwIACwoFAwQIBwIDAQEIRnYvNxgALxc8Lxc8AS4uLi4u'
    'Li4uADEwAUlouQAIAAxJaGGwQFJYOBE3uQAM/8A4WQkBIxMDMxMBIxMDMwMQ/q5Z4uJXJ/6sXejiVwIR/nABkAGJ/nf+cAGQAYkA'
    'AQBn//IDYQLrAAsAmEBHAQwMQA0ACAILCgYFBAALCwACAQIKCQoDBwMECAcIBQUGCQkKBAQJBQQFAgIDAQABBgcGBwsKCwgICQAA'
    'AQcHAAkHAQMBBEZ2LzcYAC88LzwBhy4IxAjECMQI/AjECMQIxIcuCMQIxAjECPwIxAjECMQBLi4uLi4uAC4uMTABSWi5AAQADElo'
    'YbBAUlg4ETe5AAz/wDhZJQcJAScJATcJARcBA2FM/s7+zEgBMv7QRgEzATFL/tA9SgEy/s1LATIBM0n+zQEzSf7OAAADAGkAgQND'
    'A1cACwAPABoAW0AjARsbQBwMDwwNAA4NDQYQAAIWBg0MAw8OGQMTAwMJCRMBDUZ2LzcYAC8vEP0Q/S88/TwBLzz9PBD9PBD9PAAx'
    'MAFJaLkADQAbSWhhsEBSWDgRN7kAG//AOFkBFAYjIiY1NDYzMhYBITUhARQGIyImNTQ2MzICJS8kIzMyJCMwAR79JgLa/uIvJCMz'
    'MiRTAwQkMzQjJC8w/p1T/sAkMjMjJDEAAAEAAAAAAPMFfAAKAHlAMwELC0AMAAEJCAMCAAAKAAEHAwMEAgIDCAcICQcAAAEKCgAF'
    'BAEHBgQDAwgHCgYFAAEGRnYvNxgAPzwvLzz9PAEvPP08hy4IxA78CMSHLgjEDvwIxAEuLi4uLgAuMTABSWi5AAYAC0loYbBAUlg4'
    'ETe5AAv/wDhZEwcnNyMRIxEzJzfzbBtHjiWuQhsFDm4XR/sCBSBDGQAB/y8AAAAlBXwACgB3QDEBCwtADAAFCQgGBAMICQcJCgcG'
    'BgcEAwQFBgUGBwYGBwIBAQoAAwIDCgkHAQAAAQZGdi83GAA/PC8vPP08AS88/TyHLg7ECPwIxIcuDsQI/A7EAS4uLi4uAC4xMAFJ'
    'aLkABgALSWhhsEBSWDgRN7kAC//AOFkzIxEjFwcnNxcHMyUljkUXcXEXQ7EE/kcXbm4ZQwABAH0DawHTBgUAFQBJQBkBFhZAFwAV'
    'AAoCEQQBEQYDDggDDhUOARFGdi83GAAvLxD9EP0BL/0Q/S4uADEwAUlouQARABZJaGGwQFJYOBE3uQAW/8A4WQEGBwYVFBc2MzIV'
    'FAcGIyImNTQ3NjcB01lBTCAnM2YwKz1SZmxhiQXOO1ZlXDcrL2Q7IR5lUo2Of0kAAAEAaQNoAbwGAwAUAEZAFwEVFUAWAAkFBA4C'
    'AAcBAAsDEhIEAQRGdi83GAAvLxD9AS/9EP08PAAuMTABSWi5AAQAFUloYbBAUlg4ETe5ABX/wDhZARQHBgc1NjU0JwYjIiY1NDc2'
    'MzIWAbxrW43kISE+KTkxKz1WYgVJj4p2UjiisD4nMTgpOiMfZAAAAgBGAyICPAUPABIAJQBHQBcBJiZAJwAXEwQADQIGGQIgIg8c'
    'CQEgRnYvNxgALzwvPAEv/S/9Li4uLgAxMAFJaLkAIAAmSWhhsEBSWDgRN7kAJv/AOFkBBgcGFRYVFAYjIicmNTQ3FBcWBwYHBhUW'
    'FRQGIyInJjU0NxQXFgI8DRw0Uj8tNCAdywEE9g0cNFI/LTQgHcsBBAR6BhQzLBZaLUIqJjbJnlMKKBAGFDMsFlotQiomNsmeUwoo'
    'AAACAE4DGAJEBQUAEgAlAEdAFwEmJkAnAB0ZCgYMAgATAh8iDxUCARlGdi83GAAvPC88AS/9L/0uLi4uADEwAUlouQAZACZJaGGw'
    'QFJYOBE3uQAm/8A4WQEUBzQnJic2NzY1JjU0NjMyFxYFFAc0JyYnNjc2NSY1NDYzMhcWAkTLAQQYDRw0Uj4uNh8c/vLLAQQYDRw0'
    'Uj4uNh8cBH/JnlMKKBAGFDMsFlouQSklOMmeUwooEAYUMywWWi5BKSUAAQBEAIEByQQRAAUAPUARAQYGQAcABQQDAgEABQMBBEZ2'
    'LzcYAC8vAS4uLi4uLgAxMAFJaLkABAAGSWhhsEBSWDgRN7kABv/AOFkBAxMVCQEByeTe/oEBhQPG/oL+hEsBxwHJAAEAQACBAckE'
    'FAAFAD1AEQEGBkAHAAUEAwIBAAUBAQFGdi83GAAvLwEuLi4uLi4AMTABSWi5AAEABkloYbBAUlg4ETe5AAb/wDhZCQE1EwM1Acn+'
    'd+feAkr+N0sBfgF9TQAAAQC0//4B3wG5AA8AREAWARAQQBEACwsJAAIGDQMDCQgDAAEGRnYvNxgAPy88EP0BL/0uLgAuMTABSWi5'
    'AAYAEEloYbBAUlg4ETe5ABD/wDhZJRQGIyImNTQ3MwYVNjMyFgHfTjhEYZ5nghkZM0N/N0pnRJCAcF8JQQACALT//wHfA3IACwAb'
    'AE5AHAEcHEAdDBcXFQwCEgYCAAkDAw8DGRUUAwABEkZ2LzcYAD8vPC/9EP0BL/0v/S4uAC4xMAFJaLkAEgAcSWhhsEBSWDgRN7kA'
    'HP/AOFklFAYjIiY1NDYzMhYTFAYjIiY1NDczBhU2MzIWAcRCMTFCQzAwQxtNOURhnmeCGRk0QnMxQ0MxMENDAZQ4SGZFkIBwXwhA'
    'AAIAvgABAvgEsAAlADEAXEAkATIyQDMAGAsaGRULBwIALAImEgIfDwMiBAMJLwMpIikAAR9Gdi83GAA/LxD9L/0Q/QEv/S/9L/0u'
    'Li4uAC4uMTABSWi5AB8AMkloYbBAUlg4ETe5ADL/wDhZARQHBiMiJjU0MzIXNicmIyIGFRQSFRQPASc3Ni8BJjU0NjMyFxYDFAYj'
    'IiY1NDYzMhYC+CMnPSY6ZiAkCS43XUtcxgMPMgcFS5FPsn1pT1OyNSUlNDQlJTUDwEA2PTMmcxIxND5lTE/+p18aGHkDZkdJjWSD'
    'fr5CRvwyJTQ0JSU0NAAB/+YAAQPMAQIAAwA+QBIBBARABQADAgEAAwIBAAABAUZ2LzcYAD88LzwBLi4uLgAxMAFJaLkAAQAESWhh'
    'sEBSWDgRN7kABP/AOFklIREhA8z8GgPmAQEBAAACAFUFQgHcBwgAAwAHAIZAPwEICEAJAAUDAAMAAQgBAgcHBAYGBwIBAgMHAwAB'
    'AQIAAAEEBwQFBwUGBwcEBgYHBwQDAwACBgUCAwEHAQEBRnYvNxgALy8BLxc8/Rc8hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAQAu'
    'LjEwAUlouQABAAhJaGGwQFJYOBE3uQAI/8A4WQEFNSU1BTUlAdz+eQGH/nkBhwXMimqJa4tqiQAAAgAlBVIB9gbAACYALwBkQCcB'
    'MDBAMQAuHBUHBR4cGhYRDQQCJwEAIAEsEwMYKQMkGAMJJAkBFkZ2LzcYAC8vEP0Q/RD9AS/9L/0uLi4uLi4uLgAuLi4uLjEwAUlo'
    'uQAWADBJaGGwQFJYOBE3uQAw/8A4WQEUBxYXByYnBiMiJyY1Njc2NzYjIgcnNjMyFxYHNjcmNTQ3NjMyFgc0IyIGFRQXNgH2QhEb'
    'OyougjwRNAsMMxEGDiodFCMpKkIRDh00IxdJPSIiMFMjEB4lLAZeWCkWB1cHI0ExCgUEDAQXMRUiLiwmJgsTIyNBOjA/Qy8kECcV'
    'HAAAAgBZ+6AB4P1mAAMABwCGQD8BCAhACQAFAwADAAEIAQIHBwQGBgcCAQIDBwMAAQECAAABBAcEBQcFBgcHBAYGBwcEAwMAAgYF'
    'AgMBBwEBAUZ2LzcYAC8vAS8XPP0XPIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEALi4xMAFJaLkAAQAISWhhsEBSWDgRN7kACP/A'
    'OFkBBTUlNQU1JQHg/nkBh/55AYf8KoppimqKaokAAAEAVQVCAdwGNQADAE9AHAEEBEAFAAMCAQAAAwABBwECAwMAAgIDAwEBAUZ2'
    'LzcYAC8vAYcuCMQI/AjEAS4uLi4AMTABSWi5AAEABEloYbBAUlg4ETe5AAT/wDhZAQU1JQHc/nkBhwXMimqJAAIAMQVNAfYGwAAZ'
    'ACIAU0AeASMjQCQAIQcFEQ0EAhoBABMBHw8DCRwDFxcJAQ1Gdi83GAAvLxD9EP0BL/0v/S4uLi4ALi4uMTABSWi5AA0AI0loYbBA'
    'Ulg4ETe5ACP/wDhZARQHFhcHJicGIyInJjU0NzY3JjU0NzYzMhYHNCMiBhUUFzYB9kIRGzsqLo0sGD4NJ3pBF0k9IiIwUyMQHicq'
    'Bl5YKRYHVwcjRjsMBgkCCCIjI0E6MD9DLyQQJBYhAAEAS/xvAdL9YgADAE9AHAEEBEAFAAMCAQACAQIDBwMAAQECAAABAwEBAUZ2'
    'LzcYAC8vAYcuCMQI/AjEAS4uLi4AMTABSWi5AAEABEloYbBAUlg4ETe5AAT/wDhZAQU1JQHS/nkBh/z5immKAAEARgVEAeQGkgAe'
    'AE1AGwEfH0AgABUGHAoaAQAOAwgSAwgYAwQdCAEKRnYvNxgALy8v/RD9EP0BL/0uLgAuLjEwAUlouQAKAB9JaGGwQFJYOBE3uQAf'
    '/8A4WQEUBwYjIicGIyI1NDc2MzIXFjMyNjMyFjMyNTQnNxYB5BkeNzshOjxeCxAeCwMCLycdDREoJy4LPhQGFTssNR1SeygkNEE7'
    'nHUzFDonKQAAAgCLBUYBmgaQAA4AFwBFQBcBGBhAGQAPAQAUAQcSAwsWAwQLBAEHRnYvNxgALy8Q/RD9AS/9L/0AMTABSWi5AAcA'
    'GEloYbBAUlg4ETe5ABj/wDhZARQHBiMiJjU0NzYzMhcWBzQmIyIVFDMyAZojJ0Y2SRwjMjMwOzhMITFWSAXxSS40QTU2RlgmLmMf'
    'OkQ5AP//AGgEGwIeBSsABwAzAAn+OP//AEkEUQJFBfAABwA0ABL+Yf//AIAENgGwBagABwA3/+T+Tv//AFYELQIMBioABwAw//f+'
    'Sv//AGAEUgJpBewABwAxADf+XP//AE4EIgIeBZgABwA2AAD+PP//ABb9MAHM/kAABwA1/8IBL///AAb8LwG8/iwABwAy/6IBFv//'
    'AFwENwIsBsYAJwA2AA7+UQAGADMS0gAD/+0DiwGyBlcAGQAiAD8AdEAxAUBAQEEAPjYpIQcFPRENBAIaAQAxAS07ASMTAR8vAysJ'
    'Aw8cAxczAys5AycXKwENRnYvNxgALy8v/RD9EP0v/RD9AS/9L/0v/S/9Li4uLi4ALi4uLi4uMTABSWi5AA0AQEloYbBAUlg4ETe5'
    'AED/wDhZARQHFhcHJicGIyInJjU0NzY3JjU0NzYzMhYHNCMiBhUUFzYTFAcGIyInBiMiNTQzMhcWMzI2MzIWMzI1NCc3FgGyQhEb'
    'OyoujSwXPw0nekEXSD0jIjBTIw8fJypBGB05OyE6PF45CwMCLycdDREoJy4LPhQF9VgqFQhXCCJGPAwFCQIHIyMkQjkwP0QwJQ8k'
    'FiD+pz0rNR1SfH9APJ11MxQ5JygAAwACA4sBoAZ8ABwAIAAkAKlAUwElJUAmACIgHhsTBhodIB0eCB4fJCQhIyMkHSAdHgceHyAg'
    'HR8fICEkISIHIiMkJCEjIyQjIh8DHgIkISADHQ4BChgBAAwDCBADCBYDBCQIAQpGdi83GAAvLy/9EP0Q/QEv/S/9Lxc8/Rc8hy4I'
    'xAj8CMSHLgjECPwIxIcuCMQI/AjEAS4ALi4uLi4uMTABSWi5AAoAJUloYbBAUlg4ETe5ACX/wDhZARQHBiMiJwYjIjU0MzIXFjMy'
    'NjMyFjMyNTQnNxYnBTUlNQU1JQGgGB05OyE6PF45CwMCLycdDREoJy4LPhQI/ngBiP54AYgEXT0rNR1SfH9APJ11MxQ5JyiOimqJ'
    'a4tqiv//AEwEJgJVB3IAJwA2ACD+QQAGADEi4gACAOADigJ/BcoAAwAiAG1ALQEjI0AkBBkKAyAOAwIBAAIBAgMHAwABAQIAAAEe'
    'AQQSAwwMAxYIAxwhAQEORnYvNxgALy8v/S/9EP0BL/2HLgjECPwIxAEuLi4uLi4ALi4uMTABSWi5AA4AI0loYbBAUlg4ETe5ACP/'
    'wDhZAQU1JTcUBwYjIicGIyI1NDc2MzIXFjMyNjMyFjMyNTQnNxYCdP54AYgLGR43OyI6PF4LEB4LAwIwJx0NESgnLgw/FAQUimmK'
    '0DssNR1SeyclNEE7nHUzFDonKQAD//ADigGPBosAAwAiACYApEBQAScnQCgEJhkKAwEgDgIBAgMIAwAkJCUjIyQCAQIDBwMAAQEC'
    'AAABJSQlJgcmIyQkJSMjJCUkAgMBAiYjAwMAHgEEEgMMDAMWCAMcISQBDkZ2LzcYAC8vL/0v/RD9AS/9Lxc8/Rc8hy4IxAj8CMSH'
    'LgjECPwIxIcuCMQI/AjEAS4uAC4uLi4uMTABSWi5AA4AJ0loYbBAUlg4ETe5ACf/wDhZAQU1JTcUBwYjIicGIyI1NDc2MzIXFjMy'
    'NjMyFjMyNTQnNxYDBTUlAYT+eAGICxkeNzsiOjxeCxAeCwMCMCcdDREoJy4MPxQL/ngBiATVimmK0DssNR1SeyclNEE7nHUzFDon'
    'Kf2yimmKAP//AFwF3AIsCGsABwBAAAABpf//ADEF3wItCUQABwBBAAABwP//ADoGFQIaCYIABwBC//cB9v//AJQF5gKcCTIABwBD'
    'AEgBwP//AEoFxQIaCDoABwBEAAkBwP//AEsFsQIaCTwABwBFAAABtwABAK7/8QK2AlUAHgBLQBkBHx9AIAAdEwARCAYFABkCChYD'
    'DQ0FAQVGdi83GAAvLxD9AS/9Li4uLi4ALi4uMTABSWi5AAUAH0loYbBAUlg4ETe5AB//wDhZAQcGBwYHNTY3JjU0NjMyFxYVFCMi'
    'JiMiBhUUFxYzMgK2JVKZplItWXzMVzMsNTAXWhggUi43RDABOJQPNzwxXS83N09XxBYbLj8aOx8bJi4AAAIAYP/NAqMGBAAMACcA'
    'TkAaASgoQCkNJiMXFBEKHx0ZDQcGAQwAHQQBGUZ2LzcYAC8vAS88/S4uLi4uAC4uLi4uLjEwAUlouQAZAChJaGGwQFJYOBE3uQAo'
    '/8A4WSUUBwYjIicDNzYzMhUTFAcGIyImIyIGIyI1NDc2MyIVFBcWMzI2MzIB7EsTCQ0IX7gSCAm3g3BKHGsYEC0MHiksIRJZgiwn'
    'jA4X30mgKVwEIZkPFQEJPjUtHTsuHkVLAwUTGzAAAgBgAAECuAYEABYAMQBhQCUBMjJAMwAwLSEeGw4pJyMXFgkABwITBQISExYV'
    'AwAnAQAAASNGdi83GAA/PC8Q/TwBLzz9EP0uLi4uLi4uAC4uLi4uLjEwAUlouQAjADJJaGGwQFJYOBE3uQAy/8A4WSUjIicmNTQn'
    'AicmPwE2MzAXFhUDFDsBAxQHBiMiJiMiBiMiNTQ3NjMiFRQXFjMyNjMyArh2Y0ZdDxEQBgqlHgUBBgKfPBWDcEocaxgQLQweKSwh'
    'ElmCLCeMDhcBQli1d/wBHVEcCoYYAQEb/LeOBOU+NS0dOy4eRUsDBRMbMAAAAgB6/9EB6gaNAAwALwBYQCABMDBAMQ8sIhMNCiAW'
    'FBMPBwYBDAApARklAxwcBAETRnYvNxgALy8Q/QEv/S88/S4uLi4uLgAuLi4uLjEwAUlouQATADBJaGGwQFJYOBE3uQAw/8A4WSUU'
    'BwYjIicDNzYzMhU3FgcGBwYHNTY3LgE1NDYzMhcWFRQjIiYjIgcGFRQWMzI3NgGmTBQIDAhfuBIICT8FBAQIjdMhPRwziTsdJiof'
    'DEQUGRsgRiQaMibjRqIqXQQgmQ8V7g8kJQokXEoSHgxCHTqBFxkbKBoNDxYjNw8MAAACAHEAAQJxBo0AFgA5AGtAKwE6OkA7ADYs'
    'HRcOKiAeGRcWCQAHAhMSEwIFMwEjLwMmFhUDACYBAAABHkZ2LzcYAD88LxD9PBD9AS/9L/08EP0uLi4uLi4uLgAuLi4uLjEwAUlo'
    'uQAeADpJaGGwQFJYOBE3uQA6/8A4WSUjIicmNTQnAicmPwE2MzIVFhUDFDsBAxYHBgcGByc2Ny4BNTQ2MzIXFhUUIyImIyIHBhUU'
    'FjMyNzYCcXZiRl0PEREFCqQdBgEHAp87lAQEBAiM0wEhPRsziTodJioeDUQUGRsgRiQeLz0BQli1f/QBDmAcCoYYAQEb/LeOBM4P'
    'JCUKJFxKEh4MQxw6gRcZGygaDQ8WIzcQFAADAGf+JgMMBOUAGwAmAEkAaEApAUpKQEsAPC0nQzozMC4pIxwMCBMCGwBGBCAgAxc/'
    'AzYlAwwPNgQBCEZ2LzcYAC8vLzz9EP0v/RD9AS88/S4uLi4uLi4uLi4ALi4uMTABSWi5AAgASkloYbBAUlg4ETe5AEr/wDhZJRAH'
    'BiMiJyY3Njc2NSIGIyInJjU0NzYzMhcWFScmJyYjIgYVFDMyExYHBgcGByc2Ny4BNTQ2MzIXFhUUIyImIyIHBhUUFjMyNzYDDLI4'
    'D/uUHRuyqbkWWRaIPFNDUW+KSzqbGzosIh80cDRzBAMDCY3TASI8GjSJOx0mKh8MRBQaGiBGJB0yQib++788MwoKQIeTYgIfK3Jq'
    'iKSgfJthQyUcLB5GAw8PIyQLJF1KEh4MRBs6ghcZGykbDQ8XIzYQFQAAAwBo/gMDtQTBABwAJwBKAHZAMQFLS0BMAD0uKDsxLyok'
    'HRwTDQkARAE0RwQhIQMXQAM3HBsDACYDDg0PNwUBAAABCUZ2LzcYAD88Ly8vPDz9EP08EP0v/RD9AS/9Li4uLi4uLi4uLi4ALi4u'
    'MTABSWi5AAkAS0loYbBAUlg4ETe5AEv/wDhZJSMUBwYjIicmNzY3NjUjBwYnJjU0NzYzMhcWFzMFJicmIyIGFRQzMhMWBwYHBgcn'
    'NjcuATU0NjMyFxYVFCMiJiMiBwYVFBYzMjc2A7WwrDcP/JMcGq2uuUNChT9TQ1FvcUo6E7D+vBs6LCIfNHA0cwQDBAiN0wEiPBo0'
    'iTsdJiofDEQUGRsgRiQdLz0B/cM+MwoJPoqSYgECICp0aYild12AA0QlHCweRgMPESEkDCRcShIeDEQbOoEXGRsoGg0PFiM3EBQA'
    'AAIAqf3NAhcE9gAMAC8AVEAeATAwQDEPLiQNBCIYFg8HBgEMACsBGx4DJwoVARZGdi83GAAvLy/9AS/9Lzz9Li4uLi4ALi4uLjEw'
    'AUlouQAWADBJaGGwQFJYOBE3uQAw/8A4WSUUBwYjIicDNzYzMhUTFhUUBwYHBgcnNjcuATU0NjMyFxYVFCMiJiMiBwYVFBYzMgGm'
    'TBQIDAhfuBIICW8CAgQIjNMBIjwaNIk7HSYqHwxEFBgcIEYkFuNGoipdBCCZDxX5zwYNDRMmCSRdShIfC0QbOoIXGRspGw0QFSM3'
    'AAACANL96AKDBPUAFgA5AGdAKQE6OkA7ADYXKiAeGRcWCQASEwIFMwEjJgMvLAQVFhUDAA4dAQAAAQlGdi83GAA/PC8vEP08EP0v'
    '/QEv/S/9PC4uLi4uLi4uAC4uMTABSWi5AAkAOkloYbBAUlg4ETe5ADr/wDhZJSMiJyY1NAMCJyY/ATYzMBcWFQMUOwEDFgcGBwYH'
    'JzY3LgE1NDYzMhcWFRQjIiYjIgcGFRQWMzI3NgKDdmJGXg8QEQUJpR4FAQYBnjwwBAQECIzTASE9GzOJOh0mKh4NRBQZGyBGJB4v'
    'PQFCWbRyAQEBFFocCoYYAQEb/LeO/ckPJCUKJFxKEh4MQxw6gRcZGygaDQ8WIzcQFAAAAgCu/goFYwM3ADYAWQBsQCwBWlpAWwJW'
    'TD03HgRKQD45IAIvAQwpARQjAhpTAUMIAzNPA0YlAxhGGAE+RnYvNxgALy8Q/RD9L/0BL/0v/S/9L/0uLi4uLi4ALi4uLi4uMTAB'
    'SWi5AD4AWkloYbBAUlg4ETe5AFr/wDhZARYVFCMiJyYjIgcGFRQXFhcWFxYVEAcGISARNDc2MzIVFAYVFCEyNzY1NCcmJyY1NDc2'
    'MzIXFiUWBwYHBgcnNjcuATU0NjMyFxYVFCMiJiMiBwYVFBYzMjc2BV4FBwcVNHJ9XEdMKHRoKDn0o/78/iMwODkoPQF8pJLGdKou'
    'dXaElWFCK/zHBAMDCY3TASI9GzSJOx0mKh8MRBQYHCBGJB0yQgHoGg4RCxxWQyovFwwQDiAtcP7pdk8BiU+XsTgRsy/9IS1RLA4U'
    'Ei14hrXLVDdXDyMkCyRdShIfCkQcOoIXGRspGw0QFSM3EBUAAgCx/cUFsQLfADIAVQB1QDEBVlZAVwBSSDkzGgRGPDo1MhwAJwEQ'
    'LQEIHwIWTwE/SwNCMjEDACMDFEIUAQAAARZGdi83GAA/PC8vEP0Q/TwQ/QEv/S/9L/0v/S4uLi4uLi4ALi4uLi4uMTABSWi5ABYA'
    'VkloYbBAUlg4ETe5AFb/wDhZJSMiJiMiBwYVFBcWFxYXFhUUBwYjIBE0NzYzMhUUBhUUFxYzMjc2NTQnJicmNTQ3NjsBARYHBgcG'
    'Byc2Ny4BNTQ2MzIXFhUUIyImIyIHBhUUFjMyNzYFsVY21zYuKDtWNDRxLjrgnfX+BUI/JC84pGXiTH+YaIw2aWmK8sT8cgQDAwmN'
    '0wEiPBo0iTsdJiofDEQUGhogRiQdMkIBHgwSIxUNBQUOFhw+0V1BAXNlkIoqCKMzpDciFhojEQgKEB5IbmqMAR4PIyQLJF1KEh4M'
    'RBs6ghcZGykbDQ8XIzYQFQAC/+YAAQGTBOwAEwA2AGRAJwE3N0A4FjMpGhQQJx0bFg0LBgUJAgAwASAsAyMHBgMEIwUEAAEFRnYv'
    'NxgAPzwvEP08EP0BL/0v/S4uLi4uLi4uAC4uLi4uMTABSWi5AAUAN0loYbBAUlg4ETe5ADf/wDhZARQHBisBETMyNTQnJj8BNhcW'
    'FxYTFgcGBwYHJzY3LgE1NDYzMhcWFRQjIiYjIgcGFRQWMzI3NgFvOU6iYG+QcA0NdQ8PLR4cIAQDAwmN0wEiPBo0iTsdJiofDEQU'
    'GhogRiQdMkIBiJRnjAEBNi97DhKjFRAwWFICXQ8jJAskXUoSHgxEGzqCFxkbKRsNDxcjNhAVAAAC/+YAAQLmBFIAFgA5AGtALAE6'
    'OkA7ADYsHRczKiMgHh0ZFggHAA8EBC8DJhYVCQMIAwAmBwYBAwAAAQdGdi83GAA/FzwvEP0XPBD9L/0BLi4uLi4uLi4uLi4ALi4u'
    'LjEwAUlouQAHADpJaGGwQFJYOBE3uQA6/8A4WSUjIiYnBisBETMyNzY3NjMyFxYXFjsBARYHBgcGBzU2Ny4BNTQ2MzIXFhUUIyIm'
    'IyIHBhUUFjMyNzYC5l9NjydXqZ52kjgdEAwlJAsPFy54Z/74BQQECI3TIT0bNIk7HSYqHwxEFBkbIEYkGjImAVFCkwEBQSFYQUFY'
    'IUECkg8jJgokXEoSHgxDHDqBFxkbKBoNDxYjNw8MAAABALD/0QGMBPYADAA6QBABDQ1ADgAHBgEMAAoEAQdGdi83GAAvLwEvPP0u'
    'ADEwAUlouQAHAA1JaGGwQFJYOBE3uQAN/8A4WSUUBwYjIicDNzYzMhUBjEwUCA0IX7gSCArjR6EqXQQgmQ8VAAABAK8AAQJgBPUA'
    'FgBNQBsBFxdAGAAWCQAHAhMSEwIFFhUDAA4BAAABCUZ2LzcYAD88LxD9PAEv/TwQ/S4uLgAxMAFJaLkACQAXSWhhsEBSWDgRN7kA'
    'F//AOFklIyInJjU0JwInJj8BNjMwFxYVAxQ7AQJgdmNGXQ8REAUJpR4FAQYCnzwBQli1e/gBFlgdCYYYAQEb/LeOAAIAsf5CBgkC'
    'pwAhACUAd0A0ASYmQCcAJQwkIhsOJCMkJQUlIiMjJCIiIyUkJSIFIiMkJCUjIyQRAggXAgAVAwQeIwEIRnYvNxgALy8v/QEv/S/9'
    'hy4IxAj8CMSHLgjECPwIxAEuLi4uAC4uMTABSWi5AAgAJkloYbBAUlg4ETe5ACb/wDhZARQHBiEgJyY1NDc2MzIVFAYVFBcWMyA1'
    'NCcmJzc2NzYXFgEHJzcGCY+g/iP+/YXENC8fJh+vTvsCSTMrGYMLAwULZP3Lh4iIAZLYYGw7V99jX1UqEVsZnisTYhJCOBmlDgEB'
    'DXn8qYiIiAAAAgCx/kIGuAJ2ACUAKQCGQDwBKipAKwApHgUoJiUiIBEAKCcoKQUpJicnKCYmJykoKSYFJicoKCknJygUAgslJAMA'
    'CQMYDycBAAABC0Z2LzcYAD88Ly8v/RD9PAEv/YcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLgAuLi4xMAFJaLkACwAqSWhhsEBSWDgR'
    'N7kAKv/AOFklIyInJicGBwYjIBE0NzYzMhUUBhUUFxYzIDc2NzYzMhUUBxY7AQEHJzcGuEtbSz0Vj8aCzf3gMCsfJhmaZc8BgsUf'
    'NysTIzEfijz9HIeIiAE/Mz97LB0BcWReVSoWWheJMSCSFzwvMRZpTf3IiIiIAAL/5v5CAW8CugATABcAe0A2ARgYQBkAFxYUDQsG'
    'BRYVFhcFFxQVFRYUFBUXFhcUBRQVFhYXFRUWCQIABwYDBBAVBQQAAQVGdi83GAA/PC8vEP08AS/9hy4IxAj8CMSHLgjECPwIxAEu'
    'Li4uLi4ALjEwAUlouQAFABhJaGGwQFJYOBE3uQAY/8A4WQEUBwYrAREzMjU0JyY/ATYXFhcWAwcnNwFvOU6iYG+QcA0NdQ8PLR4c'
    'JoiIiAGIlGeMAQE2L3sOEqMVEDBYUvz6iIiIAAAC/+b+PQLmAf0AFgAaAINAPAEbG0AcABoZFxYIBwAZGBkaBRoXGBgZFxcYGhka'
    'FwUXGBkZGhgYGQQEDxYVCQMIAwAPGAcGAQMAAAEHRnYvNxgAPxc8Ly8Q/Rc8EP0Bhy4IxAj8CMSHLgjECPwIxAEuLi4uLi4ALjEw'
    'AUlouQAHABtJaGGwQFJYOBE3uQAb/8A4WSUjIiYnBisBETMyNzY3NjMyFxYXFjsBAQcnNwLmX02PJ1epnnaSOB0QDCUkCw8XLnhn'
    '/tWIh4cBUUKTAQFBIVhBQVghQf3DiIiHAAAEALH91QYJAqcAIQAlACkALQDbQHABLi5ALwAtKSclIwwsKigmJCIbDiQjJCUGJSIj'
    'IyQiIiMsKywtBi0qKyssKiorKCcoKQYpJicnKCYmJyUkJSIGIiMkJCUjIyQpKCkmBiYnKCgpJycoLSwtKgYqKywsLSsrLBECCBcC'
    'AAQDFR4rAQhGdi83GAAvLy/9AS/9L/2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4u'
    'Li4uLi4ALi4uLi4uMTABSWi5AAgALkloYbBAUlg4ETe5AC7/wDhZARQHBiEgJyY1NDc2MzIVFAYVFBcWMyA1NCcmJzc2NzYXFgEH'
    'JzcPASc3AQcnNwYJj6D+I/79hcQ0Lx8mH69O+wJJMysZgwsDBQtk/lB6enqXent7AQZ7enoBkthgbDtX32NfVSoRWxmeKxNiEkI4'
    'GaUOAQENefz5enp7e3p6e/66enp7AAAEALH91Qa4AnYAJQApAC0AMQDqQHgBMjJAMwAxLSspJx4FMC4sKigmJSIgEQAoJygpBikm'
    'JycoJiYnLCssLQYtKisrLCoqKzAvMDEGMS4vLzAuLi8pKCkmBiYnKCgpJycoLSwtKgYqKywsLSsrLDEwMS4GLi8wMDEvLzAUAgsl'
    'JAMACQMYDy8BAAABC0Z2LzcYAD88Ly8v/RD9PAEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjE'
    'CPwIxAEuLi4uLi4uLi4uLgAuLi4uLi4uMTABSWi5AAsAMkloYbBAUlg4ETe5ADL/wDhZJSMiJyYnBgcGIyARNDc2MzIVFAYVFBcW'
    'MyA3Njc2MzIVFAcWOwEBByc3DwEnNwEHJzcGuEtbSz0Vj8aCzf3gMCsfJhmaZc8BgsUfNysTIzEfijz9oXp6epd6e3sBBnt6egE/'
    'Mz97LB0BcWReVSoWWheJMSCSFzwvMRZpTf4Yenp7e3p6e/66enp7AAT/5v3VAh0CugATABcAGwAfAN9AcgEgIEAhFB8bGRcVHhwa'
    'GBYUDQsGBRYVFhcGFxQVFRYUFBUaGRobBhsYGRkaGBgZHh0eHwYfHB0dHhwcHRcWFxQGFBUWFhcVFRYbGhsYBhgZGhobGRkaHx4f'
    'HAYcHR4eHx0dHgkCAAcGAwQQHQUEAAEFRnYvNxgAPzwvLxD9PAEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4I'
    'xAj8CMSHLgjECPwIxAEuLi4uLi4uLi4uAC4uLi4uMTABSWi5AAUAIEloYbBAUlg4ETe5ACD/wDhZARQHBisBETMyNTQnJj8BNhcW'
    'FxYTByc3DwEnNwEHJzcBbzlOomBvkHANDXUPDy0eHK57enqWe3p6AQZ6e3sBiJRnjAEBNi97DhKjFRAwWFL9Snp6e3t6env+unp6'
    'ewAABP/m/dUC5gH9ABYAGgAeACIA50B4ASMjQCQAIh4cGhghHx0bGRcWCAcAGRgZGgYaFxgYGRcXGB0cHR4GHhscHB0bGxwhICEi'
    'BiIfICAhHx8gGhkaFwYXGBkZGhgYGR4dHhsGGxwdHR4cHB0iISIfBh8gISEiICAhBAQPFhUJAwgDAA8gBwYBAwAAAQdGdi83GAA/'
    'FzwvLxD9FzwQ/QGHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLgAuLi4u'
    'LjEwAUlouQAHACNJaGGwQFJYOBE3uQAj/8A4WSUjIiYnBisBETMyNzY3NjMyFxYXFjsBAwcnNw8BJzcBByc3AuZfTY8nV6medpI4'
    'HRAMJSQLDxcueGeDent7l3p7ewEFenp6AVFCkwEBQSFYQUFYIUH+GHp6e3t6env+unp6ewADALH/7gYJA5EAIQAlACkAq0BTASoq'
    'QCsAJyMeDCgmJCIbDiIlIiMGIyQlJSIkJCUmKSYnBicoKSkmKCgpIyIjJAYkJSIiIyUlIicmJygGKCkmJicpKSYRAggXAgAVAwQp'
    'JQQBCEZ2LzcYAC8vPBD9AS/9L/2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEuLi4uLi4ALi4uLjEwAUlouQAIACpJ'
    'aGGwQFJYOBE3uQAq/8A4WQEUBwYhICcmNTQ3NjMyFRQGFRQXFjMgNTQnJic3Njc2FxYlByc3DwEnNwYJj6D+I/79hcQ0Lx8mH69O'
    '+wJJMysZgwsDBQtk/kF7enqXenp6AZLYYGw7V99jX1UqEVsZnisTYhJCOBmlDgEBDXn2enp6enp6egADALH/7ga4A5EAJQApAC0A'
    'ukBbAS4uQC8AKyceDwUsKigmJSIgEQAmKSYnBicoKSkmKCgpKi0qKwYrLC0tKiwsLScmJygGKCkmJicpKSYrKissBiwtKiorLS0q'
    'FAILJSQDABgDCS0pCQEAAAELRnYvNxgAPzwvLzwQ/RD9PAEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4u'
    'Li4uLgAuLi4uLjEwAUlouQALAC5JaGGwQFJYOBE3uQAu/8A4WSUjIicmJwYHBiMgETQ3NjMyFRQGFRQXFjMgNzY3NjMyFRQHFjsB'
    'AQcnNw8BJzcGuEtbSz0Vj8aCzf3gMCsfJhmaZc8BgsUfNysTIzEfijz9knt6epd6enoBPzM/eywdAXFkXlUqFloXiTEgkhc8LzEW'
    'aU0CFXp6enp6enoAA//mAAECHQRKABMAFwAbAK1AVAEcHEAdFBkVEBoYFhQNCwYFFBcUFQYVFhcXFBYWFxgbGBkGGRobGxgaGhsV'
    'FBUWBhYXFBQVFxcUGRgZGgYaGxgYGRsbGAkCAAcGAwQbFwUEAAEFRnYvNxgAPzwvPBD9PAEv/YcuCMQI/AjEhy4IxAj8CMSHLgjE'
    'CPwIxIcuCMQI/AjEAS4uLi4uLi4uAC4uLjEwAUlouQAFABxJaGGwQFJYOBE3uQAc/8A4WQEUBwYrAREzMjU0JyY/ATYXFhcWEwcn'
    'Nw8BJzcBbzlOomBvkHANDXUPDy0eHK57enqWe3p6AYiUZ4wBATYvew4SoxUQMFhSAgB6enp6enp6AAAD/+YAAQLmA5EAFgAaAB4A'
    's0BZAR8fQCAAHBgdGxkXFggHABcaFxgGGBkaGhcZGRobHhscBhwdHh4bHR0eGBcYGQYZGhcXGBoaFxwbHB0GHR4bGxweHhsPBAQW'
    'FQkDCAMAHhoHBgEDAAABB0Z2LzcYAD8XPC88EP0XPC/9AYcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4u'
    'AC4uMTABSWi5AAcAH0loYbBAUlg4ETe5AB//wDhZJSMiJicGKwERMzI3Njc2MzIXFhcWOwEDByc3DwEnNwLmX02PJ1epnnaSOB0Q'
    'DCUkCw8XLnhnm3t6epZ7enoBUUKTAQFBIVhBQVghQQIVenp6enp6egAEALH/7gYJBCQAIQAlACkALQDdQHEBLi5ALwAtKSclIx4M'
    'LCooJiQiGw4kIyQlBiUiIyMkIiIjKCcoKQYpJicnKCYmJywrLC0GLSorKywqKislJCUiBiIjJCQlIyMkKSgpJgYmJygoKScnKC0s'
    'LSoGKissLC0rKywRAggXAgAVAwQrBAEIRnYvNxgALy8Q/QEv/S/9hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjE'
    'CPwIxIcuCMQI/AjEAS4uLi4uLi4uAC4uLi4uLi4xMAFJaLkACAAuSWhhsEBSWDgRN7kALv/AOFkBFAcGISAnJjU0NzYzMhUUBhUU'
    'FxYzIDU0JyYnNzY3NhcWJTcXBz8BFwcBNxcHBgmPoP4j/v2FxDQvHyYfr077AkkzKxmDCwMFC2T8PXt6epZ7enr++nt6egGS2GBs'
    'O1ffY19VKhFbGZ4rE2ISQjgZpQ4BAQ15vnp6e3t6ensBRXt7egAEALH/7ga4BCQAJQApAC0AMQDsQHkBMjJAMwAxLSspJx4PBTAu'
    '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'
    'Li4uAC4uMTABSWi5AAgAJkloYbBAUlg4ETe5ACb/wDhZAQciBwYHBisBETMyNzY3JicmIyIGIyI1NDc2MzIEMzIVBgEHJzcEWYGE'
    'k0zHnNhRbNaDbIcsc342Nn4FKB1uiEQB7Hs0Af7ph4iIAdS5SSdhSQEBHxpEESwwZS0eNMPdCgf9G4iIiAAC/+b+bgTrAvAAJgAq'
    'AJRARQErK0AsACoRAyknJiEcEwwHBgApKCkqBSonKCgpJycoKikqJwUnKCkpKigoKR8DGiYlCAMHAwAOAxcXKAYFAQMAAAEGRnYv'
    'NxgAPxc8Ly8Q/RD9Fzwv/QGHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4ALi4uMTABSWi5AAYAK0loYbBAUlg4ETe5ACv/wDhZ'
    'JSMgEQAhIxEzMjc2NyYjIgYjIjU0NzYzMgQzMhUGDwEiBxQXFjsBAQcnNwTrxv64/rP+p1FstoNZwelwNn4FJhF0mE8Bt3pPAQZV'
    'b0lZMGqz/eqIh4cBATT+zAEBLyBwUXcjJiLqjQcDDbceTRoO/fSIiIgABACx/eAE6gLZADEANQA5AD0A50B2AT4+QD8fPTs5NzUz'
    'Lh0bDzw6ODY0MjAfEQo0MzQ1BjUyMzM0MjIzODc4OQY5Njc3ODY2Nzw7PD0GPTo7Ozw6OjszMjM0BjQ1MjIzNTUyOTg5NgY2Nzg4'
    'OTc3OD08PToGOjs8PD07OzwoAgYqAwIMAxcXAgEGRnYvNxgALy8Q/RD9AS/9hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8'
    'CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4ALi4uLi4uLi4uLjEwAUlouQAGAD5JaGGwQFJYOBE3uQA+/8A4WQEGISAnJjU0'
    'NzY3JiMiBjMiNTQ3Njc2MzIXFhcWNzYVBg8BBgcGBwYVECEyNzYzMhcWAwcnNw8BJzcBByc3BN5u/rD+/aHLsI3c3TY5gQEtHTAc'
    'RGIgj8JbnIcZAglUvWSPe8UB69R3CgQHAgRpent7l3p7ewEFent7/oKiW3PuxZ1+TS1xJSA7YiFRGCEJEBEDBQkVxAYWIFWJyv6o'
    'JAMFBwGUe3t6ent7ev67enp7AAAEALH94AV5AtkAPABAAEQASAEAQIQBSUlASgBIRkRCQD4wLiIPBUdFQ0E/PTcyJB0RBT8+P0AG'
    'QD0+Pj89PT5BREFCBkJDRERBQ0NER0ZHSAZIRUZGR0VFRj49Pj8GP0A9PT5AQD1EQ0RBBkFCQ0NEQkJDSEdIRQZFRkdHSEZGRwkC'
    'GQMCPAA8OwMACwMVHwMqKhUBAAABGUZ2LzcYAD88Ly8Q/RD9EP08AS88/S/9hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8'
    'CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4uLgAuLi4uLi4uLi4uLjEwAUlouQAZAElJaGGwQFJYOBE3uQBJ/8A4WSUjIBE0'
    'NwYHBhUQITI3NjMyFxYHBiEgJyY1NDc2NyYjIgYzIjU0NzY3NjMyFxYXFjc2FRQPASIHFBcWOwEBByc3DwEnNwEHJzcFeY/+jQWf'
    'gcMB69R3CgQHAgQMbv6w/v2hy7CN3N02OYEBLR0wHERiIY7BXJ6FHQtYTFpmMXqF/qd6e3uXent7AQV6e3sBAU0lKSJZhsv+qCQD'
    'BQcRoltz7sWdfk0tcSUgO2IhURghCRARBAcEGcQKdCMR/vB7e3p6e3t6/rt6ensABP/p/gYEXwLJACEAJQApAC0A40B0AS4uQC8g'
    'LSknJSMVLCooJiQiIBcOCQgkIyQlBiUiIyMkIiIjKCcoKQYpJicnKCYmJywrLC0GLSorKywqKislJCUiBiIjJCQlIyMkKSgpJgYm'
    'JygoKScnKC0sLSoGKissLC0rKywKCQMHEgMbGysIBwABCEZ2LzcYAD88Ly8Q/RD9PAGHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSH'
    'LgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLi4ALi4uLi4uMTABSWi5AAgALkloYbBAUlg4ETe5AC7/wDhZAQciBwYH'
    'BisBETMyNzY3JicmIyIGIyI1NDc2MzIEMzIVBgMHJzcPASc3AQcnNwRZgYSTTMec2FFs1oNshyxzfjY2fgUoHW6IRAHsezQBwnp7'
    'e5d6e3sBBXp7ewHUuUknYUkBAR8aRBEsMGUtHjTD3QoH/XB7e3p6e3t6/rt6enoAAAT/5v4GBOsC8AAmACoALgAyAPhAgQEzM0A0'
    'ADIuLCooEQMxLy0rKScmIRwTDAcGACkoKSoGKicoKCknJygtLC0uBi4rLCwtKyssMTAxMgYyLzAwMS8vMCopKicGJygpKSooKCku'
    'LS4rBissLS0uLCwtMjEyLwYvMDExMjAwMR8DGiYlCAMHAwAOAxcXMAYFAQMAAAEGRnYvNxgAPxc8Ly8Q/RD9Fzwv/QGHLgjECPwI'
    'xIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLi4uLi4ALi4uLi4uLjEwAUlouQAGADNJ'
    'aGGwQFJYOBE3uQAz/8A4WSUjIBEAISMRMzI3NjcmIyIGIyI1NDc2MzIEMzIVBg8BIgcUFxY7AQEHJzcPASc3AQcnNwTrxv64/rP+'
    'p1FstoNZwelwNn4FJhF0mE8Bt3pPAQZVb0lZMGqz/nx6e3uXent7AQV6e3sBATT+zAEBLyBwUXcjJiLqjQcDDbceTRoO/kl7e3p6'
    'e3t6/rt6enoAAQCx/eAE6gLZADEAUEAcATIyQDMfLh0bDzAfEQooAgYqAwIMAxcXAgEGRnYvNxgALy8Q/RD9AS/9Li4uLgAuLi4u'
    'MTABSWi5AAYAMkloYbBAUlg4ETe5ADL/wDhZAQYhICcmNTQ3NjcmIyIGMyI1NDc2NzYzMhcWFxY3NhUGDwEGBwYHBhUQITI3NjMy'
    'FxYE3m7+sP79ocuwjdzdNjmBAS0dMBxEYiCPwluchxkCCVS9ZI97xQHr1HcKBAcCBP6Coltz7sWdfk0tcSUgO2IhURghCRARAwUJ'
    'FcQGFiBVicr+qCQDBQcAAAEAsf3gBXkC2QA8AGlAKgE9PUA+ADAuIg8FNzIkHREFCQIZAwI8ADw7AwALAxUfAyoqFQEAAAEZRnYv'
    'NxgAPzwvLxD9EP0Q/TwBLzz9L/0uLi4uLi4ALi4uLi4xMAFJaLkAGQA9SWhhsEBSWDgRN7kAPf/AOFklIyARNDcGBwYVECEyNzYz'
    'MhcWBwYhICcmNTQ3NjcmIyIGMyI1NDc2NzYzMhcWFxY3NhUUDwEiBxQXFjsBBXmP/o0Fn4HDAevUdwoEBwIEDG7+sP79ocuwjdzd'
    'NjmBAS0dMBxEYiGOwVyehR0LWExaZjF6hQEBTSUpIlmGy/6oJAMFBxGiW3PuxZ1+TS1xJSA7YiFRGCEJEBEEBwQZxAp0IxEAAAH/'
    '6QABBF8CyQAhAExAGgEiIkAjIBUgFw4JCAoJAwcSAxsbCAcAAQhGdi83GAA/PC8Q/RD9PAEuLi4uLgAuMTABSWi5AAgAIkloYbBA'
    'Ulg4ETe5ACL/wDhZAQciBwYHBisBETMyNzY3JicmIyIGIyI1NDc2MzIEMzIVBgRZgYSTTMec2FFs1oNshyxzfjY2fgUoHW6IRAHs'
    'ezQBAdS5SSdhSQEBHxpEESwwZS0eNMPdCgcAAAH/5gABBOsC8AAmAGFAJwEnJ0AoABEDJiEcEwwHBgAfAxomJQgDBwMADgMXFwYF'
    'AQMAAAEGRnYvNxgAPxc8LxD9EP0XPC/9AS4uLi4uLi4uAC4uMTABSWi5AAYAJ0loYbBAUlg4ETe5ACf/wDhZJSMgEQAhIxEzMjc2'
    'NyYjIgYjIjU0NzYzMgQzMhUGDwEiBxQXFjsBBOvG/rj+s/6nUWy2g1nB6XA2fgUmEXSYTwG3ek8BBlVvSVkwarMBATT+zAEBLyBw'
    'UXcjJiLqjQcDDbceTRoOAAACALH94ATqBKMAMQA1AIFAOQE2NkA3HzMuHRsPNDIwHxEKMjUyMwUzNDU1MjQ0NTMyMzQFNDUyMjM1'
    'NTIoAgYqAwIMAxc1AgEGRnYvNxgALy8v/RD9AS/9hy4IxAj8CMSHLgjECPwIxAEuLi4uLi4ALi4uLi4xMAFJaLkABgA2SWhhsEBS'
    'WDgRN7kANv/AOFkBBiEgJyY1NDc2NyYjIgYzIjU0NzY3NjMyFxYXFjc2FQYPAQYHBgcGFRAhMjc2MzIXFgEHJzcE3m7+sP79ocuw'
    'jdzdNjmBAS0dMBxEYiCPwluchxkCCVS9ZI97xQHr1HcKBAcCBP6BiIiI/oKiW3PuxZ1+TS1xJSA7YiFRGCEJEBEDBQkVxAYWIFWJ'
    'yv6oJAMFBwWIiIiIAAIAsf3gBXkEowA8AEAAmkBHAUFBQEIAPjAuIg8FPz03MiQdEQU9QD0+BT4/QEA9Pz9APj0+PwU/QD09PkBA'
    'PQkCGQMCPAA8OwMACwMVHwMqQBUBAAABGUZ2LzcYAD88Ly8v/RD9EP08AS88/S/9hy4IxAj8CMSHLgjECPwIxAEuLi4uLi4uLgAu'
    'Li4uLi4xMAFJaLkAGQBBSWhhsEBSWDgRN7kAQf/AOFklIyARNDcGBwYVECEyNzYzMhcWBwYhICcmNTQ3NjcmIyIGMyI1NDc2NzYz'
    'MhcWFxY3NhUUDwEiBxQXFjsBAQcnNwV5j/6NBZ+BwwHr1HcKBAcCBAxu/rD+/aHLsI3c3TY5gQEtHTAcRGIhjsFcnoUdC1hMWmYx'
    'eoX98oiIiAEBTSUpIlmGy/6oJAMFBxGiW3PuxZ1+TS1xJSA7YiFRGCEJEBEEBwQZxAp0IxEDGYiIiAAC/+kAAQRfBKMAIQAlAH1A'
    'NwEmJkAnICMVJCIgFw4JCCIlIiMFIyQlJSIkJCUjIiMkBSQlIiIjJSUiCgkDBxIDGyUIBwABCEZ2LzcYAD88Ly/9EP08AYcuCMQI'
    '/AjEhy4IxAj8CMQBLi4uLi4uLgAuLjEwAUlouQAIACZJaGGwQFJYOBE3uQAm/8A4WQEHIgcGBwYrAREzMjc2NyYnJiMiBiMiNTQ3'
    'NjMyBDMyFQYBByc3BFmBhJNMx5zYUWzWg2yHLHN+NjZ+BSgdbohEAex7NAH+iYiIiAHUuUknYUkBAR8aRBEsMGUtHjTD3QoHAkCI'
    'iIgAAv/mAAEE6wSjACYAKgCSQEQBKytALAAoEQMpJyYhHBMMBwYAJyonKAUoKSoqJykpKignKCkFKSonJygqKicaAx8mJQgDBwMA'
    'DgMXKgYFAQMAAAEGRnYvNxgAPxc8Ly/9EP0XPC/9AYcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLgAuLi4xMAFJaLkABgArSWhh'
    'sEBSWDgRN7kAK//AOFklIyARACEjETMyNzY3JiMiBiMiNTQ3NjMyBDMyFQYPASIHFBcWOwEBByc3BOvG/rj+s/6nUWy2g1nB6XA2'
    'fgUmEXSYTwG3ek8BBlVvSVkwarP+MIeIiAEBNP7MAQEvIHBRdyMmIuqNBwMNtx5NGg4DGYiIiAABALEAAQN7A4IAHgBFQBcBHx9A'
    'IAAWBBIBAAgEAg4DAhkCAAEERnYvNxgAPy8Q/RD9AS/9Li4AMTABSWi5AAQAH0loYbBAUlg4ETe5AB//wDhZARAhIDU0NzYzMhcW'
    'FxYzMjc2NTQnJic3NjMyFxYXFgN7/jr+/AMQOCAEBRUoek1Zg4x/UTwGBwYQkmZ7AWz+lfMXGYUuPxguDxYsT3VqFuAXCl2BnAAB'
    'ALQAAQSPA3MAIABTQB4BISFAIgADIBcVCQANBAAgHwMAEwMAGgcBAAABCUZ2LzcYAD88PC8Q/RD9PBD9AS4uLi4uAC4xMAFJaLkA'
    'CQAhSWhhsEBSWDgRN7kAIf/AOFklIyInBgcGIyI1NDc2MzIXFhcWMyA1NAM3NjMyFxMWOwEEj2uPY1M0Vqb7CBg5IAECESJyARaM'
    'cQ0DBwquM1xhAbhoHjLoIyp/MkMZMWpDASmVEhv+MIkAAgCxAAEDewUKAB4AIgB4QDUBIyNAJAAgGSEfFgQfIh8gBSAhIiIfISEi'
    'IB8gIQUhIh8fICIiHxIBAAgEAg4DAiICAAEERnYvNxgAPy8Q/RD9AS/9hy4IxAj8CMSHLgjECPwIxAEuLi4uAC4uMTABSWi5AAQA'
    'I0loYbBAUlg4ETe5ACP/wDhZARAhIDU0NzYzMhcWFxYzMjc2NTQnJic3NjMyFxYXFgEHJzcDe/46/vwDEDggBAUVKHpNWYOMf1E8'
    'BgcGEJJme/7WiIeHAWz+lfMXGYUuPxguDxYsT3VqFuAXCl2BnAKFiIiHAAACALQAAQSPBPsAIAAkAIZAPAElJUAmACIaAyMhIBcV'
    'CQAhJCEiBSIjJCQhIyMkIiEiIwUjJCEhIiQkIQ0EACAfAwATAwAkBwEAAAEJRnYvNxgAPzw8LxD9EP08EP0Bhy4IxAj8CMSHLgjE'
    'CPwIxAEuLi4uLi4uAC4uLjEwAUlouQAJACVJaGGwQFJYOBE3uQAl/8A4WSUjIicGBwYjIjU0NzYzMhcWFxYzIDU0Azc2MzIXExY7'
    'AQEHJzcEj2uPY1M0Vqb7CBg5IAECESJyARaMcQ0DBwquM1xh/liIiIgBuGgeMugjKn8yQxkxakMBKZUSG/4wiQN0iIiIAAABAE7+'
    'JgL/AgcAFAA6QBABFRVAFgAQCAwBABMEAQhGdi83GAAvLwEv/S4uADEwAUlouQAIABVJaGGwQFJYOBE3uQAV/8A4WSUQBwYjICcm'
    'NzY3NjU0JyYnNzYXFgL/vjsN/vOCHBq5o7tEOTZhCQ+6Rv7ixT0zCwlCfpBzMFZIKc0TD7oAAAEAUP4DA7oB4AAZAExAGgEaGkAb'
    'AAMTCw8CGQAZGAMAFhQHAQAAAQtGdi83GAA/PC8vPBD9PAEvPP0uLgAuMTABSWi5AAsAGkloYbBAUlg4ETe5ABr/wDhZJSMiJxAH'
    'BiMgJyY3Njc2NTQnJic3NhcWOwEDuixhNbg4D/71hBoYtaq+RTM+aQkNln9XARr+0LI2MwoJQ4CPZzFUPjXUEhLMAAIATv4mAv8D'
    'ugAUABgAbUAuARkZQBoAFhMXFRAIFRgVFgUWFxgYFRcXGBYVFhcFFxgVFRYYGBUMAQAYBAEIRnYvNxgALy8BL/2HLgjECPwIxIcu'
    'CMQI/AjEAS4uLi4ALi4xMAFJaLkACAAZSWhhsEBSWDgRN7kAGf/AOFklEAcGIyAnJjc2NzY1NCcmJzc2FxYDByc3Av++Ow3+84Ic'
    'Grmju0Q5NmEJD7pxh4iIRv7ixT0zCwlCfpBzMFZIKc0TD7oB9IeHiAAAAgBQ/gMDugORABkAHQB/QDgBHh5AHwAbFhQDHBoTCxod'
    'GhsFGxwdHRocHB0bGhscBRwdGhobHR0aDwIZABkYAwAdBwEAAAELRnYvNxgAPzwvLxD9PAEvPP2HLgjECPwIxIcuCMQI/AjEAS4u'
    'Li4ALi4uLjEwAUlouQALAB5JaGGwQFJYOBE3uQAe/8A4WSUjIicQBwYjICcmNzY3NjU0JyYnNzYXFjsBAQcnNwO6LGE1uDgP/vWE'
    'Ghi1qr5FMz5pCQ2Wf1f+1IeIiAEa/tCyNjMKCUOAj2cxVD411BISzAIIiIiHAAAEAE7+JgMGBF0AFAAYABwAIADRQGoBISFAIhsg'
    'HBoYFhMfHRsZFxUQCBcWFxgGGBUWFhcVFRYbGhscBhwZGhobGRkaHx4fIAYgHR4eHx0dHhgXGBUGFRYXFxgWFhccGxwZBhkaGxsc'
    'GhobIB8gHQYdHh8fIB4eHwwBAB4EAQhGdi83GAAvLwEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSH'
    'LgjECPwIxAEuLi4uLi4uLgAuLi4uLi4xMAFJaLkACAAhSWhhsEBSWDgRN7kAIf/AOFklEAcGIyAnJjc2NzY1NCcmJzc2FxYBNxcH'
    'PwEXBwE3FwcC/747Df7zghwauaO7RDk2YQkPuv4Be3p6lnt6ev76e3p6Rv7ixT0zCwlCfpBzMFZIKc0TD7oB2np6e3t6ensBRXt7'
    'egAEAFD+AwO6BDoAGQAdACEAJQDjQHQBJiZAJwAlIR8dGxYUAyQiIB4cGhMLHBscHQYdGhsbHBoaGyAfICEGIR4fHyAeHh8kIyQl'
    'BiUiIyMkIiIjHRwdGgYaGxwcHRsbHCEgIR4GHh8gICEfHyAlJCUiBiIjJCQlIyMkDwIZABkYAwAjBwEAAAELRnYvNxgAPzwvLxD9'
    'PAEvPP2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4ALi4uLi4uLi4xMAFJ'
    'aLkACwAmSWhhsEBSWDgRN7kAJv/AOFklIyInEAcGIyAnJjc2NzY1NCcmJzc2FxY7AQE3Fwc/ARcHATcXBwO6LGE1uDgP/vWEGhi1'
    'qr5FMz5pCQ2Wf1f9WHp7e5d6e3v++3p7ewEa/tCyNjMKCUOAj2cxVD411BISzAHzenp7e3p6ewFFe3t6AAABALP95gdrAo8ARgBp'
    'QCsBR0dASAAsKgo+KRoKPAEAHQIUJQEMBgQ0GDoDBC4DCCEDEEEQCAQAARRGdi83GAA/PC8vEP0Q/RD9PC/9AS/9L/0v/S4uLi4A'
    'Li4uMTABSWi5ABQAR0loYbBAUlg4ETe5AEf/wDhZARQHBiMiJwYjIicWFRQHBiEiJyY1NDc2MzIVFAYVFBcWMzI3NjU0JyYnNzYX'
    'FjMyNzY3NjMyFxYXFjMyNTQnNzY3FhcWFxYHazlHg3NLSGYkKAe4mP7/53FbMDg5KD1xUqGegZRXSjtjCw9/eEkfBxgPIyENExAm'
    'XXBWWxEDBg4ZGSEBf4tsh4B/DjMv83Vgf2acT5exOBGzL445KTE4Wjh4ZjTKFha4PA1iPD1YGj1UKmuNGwEDEiBAVQABALP95ghH'
    'AekASABvQC8BSUlASgAtKxkLA0gqGwsAHgIVJgENBwQ1SDsvA0cDACIDEUE1EQkFAQMAAAEVRnYvNxgAPxc8Ly88EP0Q/Rc8EP0B'
    'L/0v/S4uLi4uAC4uLi4uMTABSWi5ABUASUloYbBAUlg4ETe5AEn/wDhZJSMiJwYjIicGIyInFhUUBwYhIicmNTQ3NjMyFRQGFRQX'
    'FjMyNzY1NCcmJzc2FxYzMjc2NzYzMhcWFxYzMjc2NzYzMhcWFxY7AQhHenhPTmWBLTdxJCkHuJj+/+dxWzA4OSg9cVKhnoGUV0o7'
    'YwsPgHdEHAUZECciDBMIHUpGHgMcECYhDBIPJWBoAYmJgH8OMy/zdWB/ZpxPl7E4EbMvjjkpMThaOHhmNMoWFrk8EGA8PGAQPDwH'
    'aTw8Vhk8AAH/5gABBMUCjwAzAFpAIwE0NEA1ACsODSkBACEVBAoGJwMEDw4bAwQuDQwIAwQAAQ1Gdi83GAA/FzwvEP08PBD9Lzz9'
    'PAEv/S4uLgAxMAFJaLkADQA0SWhhsEBSWDgRN7kANP/AOFkBFAcGIyInBiMiJwYrAREzMjc2NzYzMhcWFxYzMjc2NzYzMhcWFxYz'
    'MjU0Jzc2NxYXFhcWBMU4RoR8UU13gE5Qo4ukWSYPFQ8iJQ4WCiFXViQLGRAmIQ0TDCNafVVbEQMGDhkZIQF/i2yHiYmJiQEBPBdY'
    'PDxeEjw8El48PV4UPVQqa40bAQMSIEBTAAAB/+YAAQXNAekANQBcQCYBNjZANwA1Dw4ACwcDBBY1NCgcEAUPAwAuIhYODQkFAQUA'
    'AAEORnYvNxgAPxc8Lzw8EP0XPBD9PDwBLi4uLgAxMAFJaLkADgA2SWhhsEBSWDgRN7kANv/AOFklIyInBiMiJwYjIicGKwERMzI3'
    'Njc2MzIXFhcWMzI3Njc2MzIXFhcWMzI3Njc2MzIXFhcWOwEFzXJ/UE6BdlFNd4BOUKOLpFkmDxUPIiUOFgohV1YkCxkQJiENEwwj'
    'Wk0iBx0RJyEMERAmX2gBiYmJiYmJAQE8F1g8PF4SPDwSXjw8XBQ8PA1jPDxWGTwAAAQAs/3mB2sEowBGAEoATgBSAQBAhQFTU0BU'
    'AFJOTEpIQSwqClFPTUtJRz4pGgpJSElKBkpHSEhJR0dIS05LTAZMTU5OS01NTlFQUVIGUk9QUFFPT1BKSUpHBkdISUlKSEhJTk1O'
    'SwZLTE1NTkxMTVJRUk8GT1BRUVJQUFE8AQAdAhQlAQw0BAYYOgMELgMIIQMQUBAIBAABFEZ2LzcYAD88Ly8Q/RD9EP08L/0BL/0v'
    '/S/9hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4ALi4uLi4uLi4uMTAB'
    'SWi5ABQAU0loYbBAUlg4ETe5AFP/wDhZARQHBiMiJwYjIicWFRQHBiEiJyY1NDc2MzIVFAYVFBcWMzI3NjU0JyYnNzYXFjMyNzY3'
    'NjMyFxYXFjMyNTQnNzY3FhcWFxYBNxcHPwEXBwE3FwcHazlHg3NLSGYkKAe4mP7/53FbMDg5KD1xUqGegZRXSjtjCw9/eEkfBxgP'
    'IyENExAmXXBWWxEDBg4ZGSH9RHp7e5d6e3v++3p7ewF/i2yHgH8OMy/zdWB/ZpxPl7E4EbMvjjkpMThaOHhmNMoWFrg8DWI8PVga'
    'PVQqa40bAQMSIEBVAZl7e3p6e3t6AUV6enoAAAQAs/3mCEcEowBIAEwAUABUAQRAiAFVVUBWAFRQTkxKLSsZCwNTUU9NS0lIKhsL'
    'AEtKS0wGTElKSktJSUpPTk9QBlBNTk5PTU1OU1JTVAZUUVJSU1FRUkxLTEkGSUpLS0xKSktQT1BNBk1OT09QTk5PVFNUUQZRUlNT'
    'VFJSUx4CFSYBDUE1BAdIOy8DRwMAIgMRUhEJBQEDAAABFUZ2LzcYAD8XPC8vEP0Q/Rc8L/08AS/9L/2HLgjECPwIxIcuCMQI/AjE'
    'hy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMQBLi4uLi4uLi4uLi4ALi4uLi4uLi4uLjEwAUlouQAVAFVJaGGwQFJYOBE3'
    'uQBV/8A4WSUjIicGIyInBiMiJxYVFAcGISInJjU0NzYzMhUUBhUUFxYzMjc2NTQnJic3NhcWMzI3Njc2MzIXFhcWMzI3Njc2MzIX'
    'FhcWOwEBNxcHPwEXBwE3FwcIR3p4T05lgS03cSQpB7iY/v/ncVswODkoPXFSoZ6BlFdKO2MLD4B3RBwFGRAnIgwTCB1KRh4DHBAm'
    'IQwSDyVgaPxoent7l3p7e/77ent7AYmJgH8OMy/zdWB/ZpxPl7E4EbMvjjkpMThaOHhmNMoWFrk8EGA8PGAQPDwHaTw8Vhk8Alx7'
    'e3p6e3t6AUV6enoAAAT/5gABBMUEowAzADcAOwA/APFAfQFAQEBBAD87OTc1Lj48Ojg2NCsODTY1NjcGNzQ1NTY0NDU4Ozg5Bjk6'
    'Ozs4Ojo7Pj0+PwY/PD09Pjw8PTc2NzQGNDU2Njc1NTY7Ojs4Bjg5Ojo7OTk6Pz4/PAY8PT4+Pz09PikBACEVBAoGJwMEDw4bAwQ9'
    'DQwIAwQAAQ1Gdi83GAA/FzwvEP08PBD9Lzz9PAEv/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjE'
    'CPwIxAEuLi4uLi4uLi4ALi4uLi4uMTABSWi5AA0AQEloYbBAUlg4ETe5AED/wDhZARQHBiMiJwYjIicGKwERMzI3Njc2MzIXFhcW'
    'MzI3Njc2MzIXFhcWMzI1NCc3NjcWFxYXFgE3Fwc/ARcHATcXBwTFOEaEfFFNd4BOUKOLpFkmDxUPIiUOFgohV1YkCxkQJiENEwwj'
    'Wn1VWxEDBg4ZGSH9IHt6epZ7enr++nt6egF/i2yHiYmJiQEBPBdYPDxeEjw8El48PV4UPVQqa40bAQMSIEBTAZd7e3p6e3t6AUV6'
    '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'
    'NDc2MzIXFhUUBxY7ASU0JiMiBwYVFBYzMjc2AwcnNwMrQ2NwWDbIk0ZuXidBTlpxYTkxShSDYv7nUykvLzNlIRwxOgKIiIgBKyIo'
    'dQEBD1pBY3WIX1FtkE4PtylUIyUtIVwhJwK2iIiHAAAEALP+BgTnA+EAKAAzADcAOwC/QF8BPDxAPQA5NTIKOjg2NDApDDQ3NDUG'
    'NTY3NzQ2Njc4Ozg5Bjk6Ozs4Ojo7NTQ1NgY2NzQ0NTc3NDk4OToGOjs4ODk7OzggAgAPAgYXASgALQMkEwMCOzcCHAAZAAEGRnYv'
    'NxgAPz8vLzwQ/S/9AS88/S/9EP2HLgjECPwIxIcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxAEuLi4uLi4uAC4uLi4xMAFJaLkABgA8'
    'SWhhsEBSWDgRN7kAPP/AOFkFECEiJyY1NDc2MzIVFAYVFBcWFxY3NjU0JyIGIyInJjU0NzYzMhcWFScmJyYjIgYVFDMyEwcnNw8B'
    'JzcE5/1/53FbMDg5KD1xUaKVjL4MG2obgTtQPUtxhlRKqBs6LCMgM4gzsHp7e5d6e3si/ih/ZptPl7E3ErIvjjkpAQEnNWwbHgMc'
    'JmSBhKOVg6VjQyUcKx9GAn96ent7enp7AAQAs/4GBdMD4QAsADcAOwA/AM1AZwFAQEBBAD05DT48Ojg0LSwjDwA4Ozg5Bjk6Ozs4'
    'Ojo7PD88PQY9Pj8/PD4+Pzk4OToGOjs4ODk7Ozg9PD0+Bj4/PDw9Pz88EgIJGgEBAzEDJywrAwAWAwU2AwA/OwUfHAEDAAABCUZ2'
    'LzcYAD8XPC8vPBD9EP0Q/Twv/QEvPP0v/YcuCMQI/AjEhy4IxAj8CMSHLgjECPwIxIcuCMQI/AjEAS4uLi4uLi4uLi4ALi4uMTAB'
    'SWi5AAkAQEloYbBAUlg4ETe5AED/wDhZJSMWFRAhIicmNTQ3NjMyFRQGFRQXFhcWNzY1NCciBiMiJyY1NDc2MzIXFhczBSYnJiMi'
    'BhUUMzITByc3DwEnNwXT7gL9f+dxWzA4OSg9cVGilYy+DBtqG4E7UD1LcX1SOwn9/mwbOiwjIDOIM6t6e3uXe3p6AR0b/j1/ZptP'
    'l7E3ErIvjjkpAQEnNWwbHgMcJmSBhKOIYmEPQyUcKx9GAn96ent7enp7AAAE/+YAAQJ7BPkAGAAjACcAKwDAQGABLCxALQApJSoo'
    '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'
    'FBkbIEYkEAPeBAMFCv79uXM0cUyXNyUI8z9ytcxGAwIJyR5BmxICAQmO/fJIrbpHcqQLEZ38YjopQy0LAgMFLg8kJQokXEoSHg1B'
    'HTqBFxkbKBoNDxYjNwACABn/0AUTBo0AMQBSAHlAMgFTU0BUAFFHODIrJB4WFQVFOzk0MSgaCwARAi0mAi4tTgE+SgNBMTADAEEJ'
    'AQAAATlGdi83GAA/PC8vEP08EP0BL/0vPP0Q/S4uLi4uLi4uLgAuLi4uLi4uLi4uMTABSWi5ADkAU0loYbBAUlg4ETe5AFP/wDhZ'
    'JSMiJyYnAgcGBQY1NDc2NzY1NCcmJwcmJyY1NDc2NzYXBBcWFzYRNCc3NjMyFREUOwEBFgcGBwYHJzY3LgE1NDYzMhcWFRQjIiYj'
    'IgcGFRQWMzIFEz20RjIIeosW/mtMP6+Im2h9ows3Qk8aBQQFEAFMpnARjwaXEgUGpCv8cQUEBAiM0wEhPR0xiDsdJiofDEQUGRsg'
    'RiQQAYZf2f6xcxIXBAgIHlRbaDArgZtvVTNLWhhcdBUCAQvt5Jp7rQGLT1J3Dhr8zZwEzQ8kJQokXEoSHg1BHTqBFxkbKBoNDxYj'
    'NwAAAwBO/bMEUgUAAC4ANgBZAHJALAFaWkBbAlZMNzUxJSEZFxAGBFNKQ0A+PTkzLyooJx0VEgwKAkYDTys9AT1Gdi83GAAvLy/9'
    'AS4uLi4uLi4uLi4uLi4uLi4uLgAuLi4uLi4uLi4uLi4xMAFJaLkAPQBaSWhhsEBSWDgRN7kAWv/AOFkBFhUUIyInEAcGBxYVFAcG'
    'IyInNzY3JicGByYnJjU0NzY3FhcWATY9ATQnNzYXFgEmJwYHFjc2ARYHBgcGBzU2Ny4BNTQ2MzIXFhUUIyImIyIHBhUUFjMyNzYE'
    'UAIOFDBSMz1EbEnwUTpNim2P2BYFBQebMAUEBAu8AYhOHqgLBjb++Q8rVUUmQkj+zAUEBAiN0yE9GzSJOx0mKh8MRBQZGyBGJBoy'
    'JgPeBAMFCv79uXM0cUyXNyUI8z9ytcxGAwIJyR5BmxICAQmO/fJIrbpHcqQLEZ38YjopQy0LAgP99A8jJgokXEoSHgxDHDqBFxkb'
    'KBoNDxYjNw8MAAIATv2zBRME6wAxAFQAeEAxAVVVQFYAUUcyJB4WFQkFTkU+Ozk4NDEoGgsAEQItJgIuLUEDSjEwAwArOAEAAAE4'
    'RnYvNxgAPzwvLxD9PC/9AS88/RD9Li4uLi4uLi4uLi4uAC4uLi4uLi4uLjEwAUlouQA4AFVJaGGwQFJYOBE3uQBV/8A4WSUjIicm'
    'JwIHBgUGNTQ3Njc2NTQnJicHJicmNTQ3Njc2FwQXFhc2ETQnNzYzMhURFDsBARYHBgcGBzU2Ny4BNTQ2MzIXFhUUIyImIyIHBhUU'
    'FjMyNzYFEz20RjIIeosW/mtMP6+Im2h9ows3Qk8aBQQFEAFMpnARjwaXEgUGpCv8pgUEBAiN0yE9GzSJOx0mKh8MRBQZGyBGJBoy'
    'JgGGX9n+sXMSFwQICB5UW2gwK4Gbb1UzS1oYXHQVAgEL7eSae60Bi09Sdw4a/M2c/ZMPIyYKJFxKEh4MQxw6gRcZGygaDQ8WIzcP'
    'DAAAAgBt/8kEUgUAAC4ANgBXQB4BNzdAOAI1MSUhGRcGBDMvKignHRUSDAoCKxABHUZ2LzcYAC8vAS4uLi4uLi4uLi4uAC4uLi4u'
    'Li4uMTABSWi5AB0AN0loYbBAUlg4ETe5ADf/wDhZARYVFCMiJxAHBgcWFRQHBiMiJzc2NyYnBgcmJyY1NDc2NxYXFgE2PQE0Jzc2'
    'FxYBJicGBxY3NgRQAg4UMFIzPURsSfBROk2KbY/YFgUFB5swBQQEC7wBiE4eqAsGNv75DytVRSZCSAPeBAMFCv79uXM0cUyXNyUI'
    '8z9ytcxGAwIJyR5BmxICAQmO/fJIrbpHcqQLEZ38YjopQy0LAgMAAAEAXP/QBRME6wAxAF1AIwEyMkAzACQeFhUFMSgaCwARAi0m'
    'Ai4tMTADACsJAQAAARpGdi83GAA/PC8vEP08AS88/RD9Li4uLi4ALi4uLi4xMAFJaLkAGgAySWhhsEBSWDgRN7kAMv/AOFklIyIn'
    'JicCBwYFBjU0NzY3NjU0JyYnByYnJjU0NzY3NhcEFxYXNhE0Jzc2MzIVERQ7AQUTPbRGMgh6ixb+a0w/r4ibaH2jCzdCTxoFBAUQ'
    'AUymcBGPBpcSBQakKwGGX9n+sXMSFwQICB5UW2gwK4Gbb1UzS1oYXHQVAgEL7eSae60Bi09Sdw4a/M2cAAEAs/4KBWMCrAA2AFJA'
    'HgE3N0A4Ah4EIAIvAQwpARQjAholAxgIAzMzGAEaRnYvNxgALy8Q/RD9AS/9L/0v/S4uAC4uMTABSWi5ABoAN0loYbBAUlg4ETe5'
    'ADf/wDhZARYVFCMiJyYjIgcGFRQXFhcWFxYVEAcGISARNDc2MzIVFAYVFCEyNzY1NCcmJyY1NDc2MzIXFgVeBQcHFTRyfVxHTCh0'
    'aCg59KP+/P4jMDg5KD0BfKSSxnSqLnV2hJVhQisB6BoOEQscVkMqLxcMEA4gLXD+6XZPAYlPl7E4EbMv/SEtUSwOFBIteIa1y1Q3'
    'AAEAsf3FBbEBAgAyAFZAIAEzM0A0ABoEMhwAJwEQLQEIHwIWIwMUMjEUAQAAARZGdi83GAA/PC8vPBD9AS/9L/0v/S4uLgAuLjEw'
    'AUlouQAWADNJaGGwQFJYOBE3uQAz/8A4WSUjIiYjIgcGFRQXFhcWFxYVFAcGIyARNDc2MzIVFAYVFBcWMzI3NjU0JyYnJjU0NzY7'
    'AQWxVjbXNi4oO1Y0NHEuOuCd9f4FQj8kLzikZeJMf5hojDZpaYryxAEeDBIjFQ0FBQ4WHD7RXUEBc2WQiioIozOkNyIWGiMRCAoQ'
    'HkhuaowABAC2//AGYgWQAAgAJwBhAGgAo0BJAWlpQGooY15TRjQuJh4PBGViW1JDQjwlEwhQUgVSU1VTU1UEBgECTgIoWQIoIwEJ'
    'ZwM4TFcDLA0DIRsDEREDF2IDOAAsMAE8RnYvNxgALzwvL/0v/RD9L/0Q/TwQ/QEv/S/9EP0v/TyHLg7ECPwOxAEuLi4uLi4uLi4u'
    'AC4uLi4uLi4uLi4xMAFJaLkAPABpSWhhsEBSWDgRN7kAaf/AOFkBFhUUIzY1NCcTFAcGIyInBiMiNTQ3NjMyFxYzMjYzMhYzMjU0'
    'JzcWARQHBiMiJwYjIicmJwYHBiMiJyY1NDc2NzY3Jzc2MzIXEhcWMzI1NCcmJzcSFxYzMjU0Azc2MzIXEgUnBgcWMzIEORc+ASLz'
    'FxsyNB8zN1ULDxoKAwIqIxoMDyQjKQo4EwF+bElptUYvnIZGISUdNhI7Z0tkIkVmMG0MdiEFBgUmFSl5XwINIrwqFiRjWFGjDAcK'
    'BU378SHDHwl0PgWQM0CBDw9TQv7bNSgwGkpvIiIvPTKNai4TMiQk/NnJWTyYmVAmYTU+BxYdOV8uXUIfK2dgGyj+vD16fQ0Ugv5u'
    '/lpXjoFPAYuCDRr+R+WgRkgYAAABAFUCiAFGA3kACwA2QA4BDAxADQAAAgYJAwEGRnYvNxgALy8BL/0AMTABSWi5AAYADEloYbBA'
    'Ulg4ETe5AAz/wDhZARQGIyImNTQ2MzIWAUZHMjFHRTEzSAMAMkZHMTFIRgAAAAAQAAAA3AkJBQACAgIFAwMFBAIEAgMDAwQFBQUE'
    'BQUEAgQDAwQEBAQEBAAAAAACAgMDAgICAgMEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMDAwIDAwQCAwYGAgMCAwcHAgMH'
    'BwIDBwcCAwcHAgMGBgUFBgYFBQYGBQUGBgUFBAUEBQMEAwQDBAgJBQYICQUGCAkGBggJBgYGBgQFBgYEBQUFBAMFBQQDBwcDAwYG'
    'AwMICQMFCAkDBQUGAgMEBQMEBQYCAwMEAwQEBAMEBgYCAwUGBQYFBgUGBgYHAgAKCgUAAgICBQMDBQUCBQIDBAMFBgUFBQYGBQIF'
    'AwMFBQQEBQUAAAAAAgIDAwMDAgIEBQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAwMCAwQEAgMHBwIDAgMICAIDCAgDAwgI'
    'AwMICAMDBgcFBgYHBQYGBwUGBgcFBgQGBAYEBQQFBAUJCgYHCQoGBwkKBgcJCgYHBwcFBQcHBQUGBgQEBgYEBAcIAwQGBwMECQoE'
    'BQgKBAUGBwIDBQYDBAYHAgMEBAQFBAUEBQcHAwMFBgUGBQYFBgcHCAIACwsGAAICAgYEAwYFAwUCBAQEBQYGBgUGBgUCBQMDBQUF'
    'BQUFAAAAAAMDAwMDAwMDBAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEAwMEBQMDBwgCBAIDCAkCBAgJAwQICQMECAkD'
    'BAcHBgcHBwYHBwcGBwcHBgcFBgUGBAUEBQQFCgsHCAoLBwgKCwcHCgsHBwcIBQYHCAUGBgcFBAYHBQQICQMEBwgDBAkLBAYJCwQG'
    'BwcCAwUGBAUHBwIEBAUEBQUFBAUHCAMEBgcGBwYHBgcHCAkCAAwMBgADAwIGBAQGBgMGAgQEBAYHBgYGBwcGAgYEBAYGBQUGBgAA'
    'AAADAwQEAwMDAwQFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQEBAMEBQUDBAgIAgQCAwkKAgQJCgMECQoDBAkKAwQHCAcH'
    'BwgHBwcIBwcHCAcHBQcFBwUFBQUFBQsMBwkLDAcJCwwHCAsMBwgICAYHCAgGBwcHBQQHBwUECQoEBQcJBAUKDAUGCgwFBgcIAgQG'
    'BwQFBwgCBAUFBAYFBQQGCAgDBAcHBwcHBwcHCAgKAgANDQcAAwMDBwQEBwYDBgMEBQQGBwcHBgcHBgMGBAQGBgUFBgYAAAAAAwME'
    'BAMDAwMFBgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEBAQDBAUGAwQJCQIFAwQKCwIFCgsDBQoLAwUKCwMFCAkHCAgJBwgI'
    'CQcICAkHCAYHBgcFBgUGBQYMDQgJDA0ICQwNCAkMDQgJCQkGBwkJBgcICAYFCAgGBQoLBAUICQQFCw0FBwsNBQcICQIEBgcEBggJ'
    'AgUFBgUGBgYFBgkJAwUHCAcIBwgHCAkJCgMADg4HAAMDAwcFBAgHAwcDBAUFBwgHCAcICAcDBwQEBwcGBgcGAAAAAAMDBAQEBAMD'
    'BQYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABQUFAwQFBgMECQoDBQMECwwDBQsMBAULDAQFCwwEBQkJCAgJCQgICQkICAkJ'
    'CAgGCAYIBQYFBgUGDQ4ICg0OCAoNDgkJDQ4JCQkKBwgJCgcICAgGBQgIBgUKCwQFCQoEBQwOBQcMDgUHCAkCBAcIBQYJCQMFBQYF'
    'BwYGBQcJCgQFCAkICQgJCAkJCgsDAA8PCAADAwMIBQQIBwQHAwUFBQcICAgHCAkHAwcFBQcHBgYHBwAAAAAEBAUFBAQEBAYHAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUFBQQEBgcEBQoKAwUDBAsMAwULDAQFCwwEBQsMBAUJCggJCQoICQkKCAkJCggJBwgH'
    'CAYHBgcGBw4PCQsODwkLDg8JCg4PCQoKCwcICgsHCAkJBwYJCQcGCwwFBgkLBQYNDwYIDQ8GCAkKAwUHCAUHCQoDBQYHBgcHBwYH'
    'CgoEBQgJCAkICQgJCgoMAwAQEAgAAwMDCAUFCQgECAMFBgYICQgJCAkJCAMIBQUICAcHCAcAAAAABAQFBQQEBAQGBwAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAFBQUEBQYHBAULCwMGAwUMDQMGDA0EBgwNBAYMDQQGCgsJCgoLCQoKCwkKCgsJCgcJBwkGBwYH'
    'BgcPEAoLDxAKCw8QCgsPEAoLCgsICQoLCAkJCgcGCQoHBgwNBQYKCwUGDhAGCA4QBggJCgMFCAkGBwoLAwYGBwYIBwcGCAsLBAYJ'
    'CgkKCQoJCgsLDQMAEREJAAQEAwkGBQkIBAgDBQYGCAoJCQgJCggDCAUFCAgHBwgIAAAAAAQEBQUEBAQEBggAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAABgYGBAUGCAQFCwwDBgMFDQ4DBg0OBAYNDgQGDQ4EBgoLCQoKCwkKCgsJCgoLCQoHCQcJBggGCAYIEBEK'
    'DBARCgwQEQsLEBELCwsMCAkLDAgJCgoIBgoKCAYNDgUHCgwFBw4RBgkOEQYJCgsDBQgKBggKCwMGBggGCAcIBggLDAQGCQsJCwkL'
    'CQsLDA4DABISCQAEBAQJBgUKCQQJBAYGBgkKCQoJCgoJBAkGBgkJCAgJCAAAAAAEBAYGBQUEBAcIAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAYGBgQFBwgEBQwNAwYDBQ4PAwYODwUGDg8FBg4PBQYLDAoLCwwKCwsMCgsLDAoLCAoICgcIBwgHCBESCw0REgsN'
    'ERILDBESCwwMDQkKDA0JCgoLCAcKCwgHDQ8GBwsNBgcPEgcKDxIHCQsMAwUJCgYICwwDBgcIBwkICAcJDA0FBgoLCgsKCwoLDA0O'
    'BAATEwoABAQECgYGCgkECQQGBwcJCwoKCQoLCQQJBgYJCQgICQkAAAAABQUGBgUFBAQHCQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAGBgYEBgcJBAYNDQMHBAUOEAMHDhAFBw4QBQcOEAUHDA0KCwwNCgsMDQoLDA0KCwgLCAsHCQcJBwkSEwsOEhMLDhITDA0S'
    'EwwNDA0JCgwNCQoLCwgHCwsIBw4QBgcMDgYHEBMHChATBwoLDAMGCQsHCAwNAwcHCQcJCAkHCQ0NBQcKDAoMCgwKDA0NDwQAFBQK'
    'AAQEBAoHBgsKBQoEBgcHCgsKCwoLCwoECQYGCgoICAoJAAAAAAUFBgYFBQUFBwkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'BwcHBQYICQUGDQ4EBwQGDxEEBw8RBQcPEQUHDxEFBwwNCwwMDQsMDA0LDAwNCwwJCwkLCAkICQgJExQMDhMUDA4TFAwNExQMDQ0O'
    'CgsNDgoLDAwJBwwMCQcPEAYIDA4GCBEUCAsRFAgKDA0EBgkLBwkMDQQHCAkHCgkJBwoNDgUHCwwLDAsMCwwNDhAEABUVCwAFBQQL'
    'BwYLCgUKBAcIBwoMCwsKDAwKBAoHBwoKCQkKCgAAAAAFBQcHBQUFBQgKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcHBwUG'
    'CAkFBg4PBAcEBhARBAcQEQYHEBEFBxARBQcNDgsNDQ4LDQ0OCw0NDgsNCQwJDAgKCAoIChQVDQ8UFQ0PFBUNDhQVDQ4ODwoLDg8K'
    'CwwNCQgMDQkIEBEHCA0PBwgSFQgLEhUICwwOBAYKDAcJDQ4EBwgJCAoJCggKDg8FBwsNCw0LDQsNDg8RBAAWFgsABQUECwcGDAsF'
    'CwQHCAgLDAsMCwwMCwQKBwcLCwkJCgoAAAAABQUHBwYGBQUICgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHBwcFBggKBQcP'
    'DwQIBAYREgQIERIGCBESBggREgYIDg8MDQ4PDA0ODwwNDg8MDQoMCgwICggKCAoUFg0QFBYNEBUWDg8VFg4PDg8LDA4PCwwNDQoI'
    'DQ0KCBASBwgOEAcIExYIDBMWCAsNDgQHCgwICg0PBAgICggLCgoICw8PBggMDgwODA4MDg8PEgQAFxcMAAUFBAwIBwwLBQsFBwgI'
    'Cw0MDAsNDQsFCwcHCwsKCgsLAAAAAAYGBwcGBgUFCQsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAgIBQcJCgUHEBAECAQH'
    'ERMECBETBggREwYIERMGCA4PDQ4ODw0ODg8NDg4PDQ4KDQoNCQoJCgkKFRcOEBUXDhAWFw4PFhcODw8QCw0PEAsNDQ4KCQ0OCgkR'
    'EwcJDhAHCRQXCQwTFwkMDg8EBwsNCAoODwQICQoJCwoKCQsQEAYIDA4MDgwODA4QEBIFABgZDAAFBQUMCAcNDAYMBQgJCAwODA0M'
    'DQ4MBQsICAsLCgoLCwAAAAAGBggIBgYGBgkLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAYHCQsGBxARBAgFBxIUBAgS'
    'FAYIEhQGCBIUBggPEA0ODxANDg8QDQ4PEA0OCg0KDQkLCQsJCxYZDhEWGQ4RFxgPEBcYDxAQEQwNEBEMDQ4OCwkODgsJEhQHCQ8R'
    'BwkUGAkNFBgJDA4QBAcLDggLDxAECAkLCQwKCwkMEBEGCA0PDQ8NDw0PEBETBQAAAQAAAAEAAJufK0pfDzz1AAAIAAAAAAC2W95N'
    'AAAAALZb39L/L/ugCEcJggABAAMAAgABAAAAAAABAAAFPf09AAAIK/8v/WQIRwABAAAAAAAAAAAAAAAAAAAA2QQAAIAAAAAAAbgA'
    'AAG4AAABjwC+BAcAewKdAPICWQC1BFAAsAP1AHIB4QC2A/UAcgGbALICkACGAt4AewLBAIwD4wBsBIAAZQQJAGsETQB5A/AAcARo'
    'AG4EiwCQA+MAdQGaALIDyQB2AoAA1gKAAIcD1QEfA9UAlgNWAEYDVgAyA80AZwOyAGkAJQAAACUAAAAlAAAAJf8vAesAfQHrAGkC'
    'igBGAooATgIAAEQCAABAAeEAtAHkALQC/AC+A6f/5gAAAFUAAAAlAAAAWQAAAFUAAAAxAAAASwAAAEYAAACLAAAAaAAAAEkAAACA'
    'AAAAVgAAAGAAAABOAAAAFgAAAAYAAABcAAD/7QAAAAIAAABMAAAA4AAA//AAAABcAAAAMQAAADoAAACUAAAASgAAAEsCtgCuAq0A'
    'YAKdAGAB5AB6AlcAcQMOAGcDmABoAeQAqQJpANIFZQCuBZcAsQFy/+YCy//mAY4AsAJFAK8GCwCxBp0AsQFy/+YCy//mBgsAsQad'
    'ALECGf/mAsv/5gYLALEGnQCxAhb/5gLL/+YGCwCxBp0AsQII/+YCy//mBOkAsQVeALEEYP/pBNL/5gTpALEFXgCxBGD/6QTS/+YE'
    '6QCxBV4AsQRg/+kE0v/mBOkAsQVeALEEYP/pBNL/5gN/ALEEdQC0A38AsQR1ALQDAQBOA6AAUAMBAE4DoABQAwEATgOgAFAHbgCz'
    'CCsAswTI/+YFsv/mB24AswgrALMEyP/mBbL/5geAALMH7wCzBPT/5gVh/+YHgACzB+8AswT0/+YFYf/mBT8ArgWiAK4D+f/kBF3/'
    '5AU/AK4FogCuA/n/5ARd/+QEowCxBMkAsQOK/+YC/v/mBKMAsQTJALEDiv/mAv7/5gX0ALEGkgCxAnz/5gMR/+YE7ACzBbsAswJ8'
    '/+YDEf/mBtAAsQfsALEDCf/mBDv/5gbIALEH7ACxAwn/5gQh/+UEuwCvBT4ArwFn/+YCaP/mA8kAsQSCALECwv/mA43/5gTcALMF'
    'YwCzAXL/5gLL/+YDDgBnA5gAaAL5ALED1gCxA33/5gOm/+YC+QCxA9YAsQVlALMFlwCxAhb/5gLL/+YEVv/uBPn/7gRWABkE+QAZ'
    'BFYATgT5AE4EVgBtBPkAXAVlALMFlwCxBmEAtgGVAFUAAAAAAAAAfgAAAH4AAAB+AAAAfgAAASQAAAHeAAACaAAAAvYAAARwAAAF'
    'BgAABXwAAAXUAAAGOAAABqYAAAccAAAHlAAACD4AAAkiAAAJ/AAACuQAAAu8AAAMYAAADP4AAA2qAAAOOgAADrQAAA8wAAAPqgAA'
    'EIIAABFaAAAR5AAAEmwAABNGAAAT/AAAE/wAABP8AAAUogAAFUQAABXWAAAWYgAAFx4AABfcAAAYQAAAGKIAABkcAAAZwAAAGq4A'
    'ABsIAAAbugAAHLAAAB1iAAAdzgAAHo4AAB76AAAfpAAAIDYAACBIAAAgWgAAIGwAACB+AAAgkAAAIKIAACC0AAAgxgAAIN4AACII'
    'AAAjJAAAIzwAACQUAAAlNAAAJUYAACVYAAAlagAAJXwAACWOAAAloAAAJkoAACcMAAAn/AAAKOIAACn2AAArMgAALIAAAC1iAAAu'
    'dAAAL9wAADFAAAAySAAAM1wAADPIAAA0YAAANVIAADZYAAA3KAAAOAYAADl6AAA7AgAAPFQAAD2yAAA+5AAAQCwAAEE8AABCVgAA'
    'Q8oAAEVUAABGpgAASAQAAEkuAABKigAAS4AAAEyYAABORAAAUCIAAFGaAABTNAAAVBwAAFU2AABV6gAAVsAAAFfoAABZQgAAWjYA'
    'AFtMAABb9AAAXKwAAF2YAABelAAAXxoAAF+8AABghAAAYWoAAGK0AABkHAAAZU4AAGaIAABnfAAAaHQAAGpsAABsagAAbiIAAG/c'
    'AABw7AAAchwAAHL8AABz+gAAdUoAAHa6AAB32gAAeRgAAHoEAAB7CAAAe+wAAHzsAAB+GgAAf2AAAICGAACByAAAgsgAAIOmAACE'
    'eAAAhS4AAIZwAACHkgAAiKQAAImcAACKzAAAjAgAAI0QAACOLgAAj54AAJEoAACScAAAk84AAJSsAACVtAAAlmYAAJc2AACYNgAA'
    'mUoAAJoiAACbHAAAm+QAAJy6AACdagAAnhIAAJ8MAACgFgAAoNQAAKGyAACipgAAo8IAAKSSAAClbgAApjAAAKcCAACnpAAAqFQA'
    'AKloAACqVgAAq3QAAKyeAACtjgAArnQAAK+EAACwoAAAsf4AALNOAAC0yAAAtjQAALeyAAC5IgAAuigAALseAAC8DgAAvPQAAL7E'
    'AAC/KAABAAAA2QCFAAkATwADAAIACABAAAoAAACMAQQAAgACAAAAGAEmAAAAAAAAAAAALAAWAAAAAAAAAAEAEgBLAAAAAAAAAAIA'
    'CABhAAAAAAAAAAMAJAClAAAAAAAAAAQAHAB3AAAAAAAAAAUAMgDiAAAAAAAAAAYAGAEgAAAAAAAAAAcAOAFUAAEAAAAAAAAAFgAA'
    'AAEAAAAAAAEACQBCAAEAAAAAAAIABABdAAEAAAAAAAMAEgCTAAEAAAAAAAQADgBpAAEAAAAAAAUAGQDJAAEAAAAAAAYADAEUAAEA'
    'AAAAAAcAHAE4AAMAAQQJAAAALAAWAAMAAQQJAAEAEgBLAAMAAQQJAAIACABhAAMAAQQJAAMAJAClAAMAAQQJAAQAHAB3AAMAAQQJ'
    'AAUAMgDiAAMAAQQJAAYAGAEgAAMAAQQJAAcAOAFUKGMpIDIwMDAgQm9ybmEgUmF5YW5laAAoAGMAKQAgADIAMAAwADAAIABCAG8A'
    'cgBuAGEAIABSAGEAeQBhAG4AZQBoQiBOYXphbmluAEIAIABOAGEAegBhAG4AaQBuQm9sZABCAG8AbABkQiBOYXphbmluIEJvbGQA'
    'QgAgAE4AYQB6AGEAbgBpAG4AIABCAG8AbABkQm9ybmEgTmF6YW5pbiBCb2xkAEIAbwByAG4AYQAgAE4AYQB6AGEAbgBpAG4AIABC'
    'AG8AbABkVmVyc2lvbiAyLjAxIC0gQnVpbGQgMTM3OQBWAGUAcgBzAGkAbwBuACAAMgAuADAAMQAgAC0AIABCAHUAaQBsAGQAIAAx'
    'ADMANwA5Qk5hemFuaW5Cb2xkAEIATgBhAHoAYQBuAGkAbgBCAG8AbABkUGFyc2EgMjAwMaggLSBCb3JuYSBSYXlhbmVoqABQAGEA'
    'cgBzAGEAIAAyADAAMAAxAK4AIAAtACAAQgBvAHIAbgBhACAAUgBhAHkAYQBuAGUAaACuAAAAAgAAAAAAAP0wAGQAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAADZAAAAAQACAAMABAAIAAsADAANAA4ADwAQABEAEgATABQAFQAWABcAGAAZABoAGwAcAB0AIAA+AEAAXgBgAKkA'
    'qgDwALgBAgEDAQQBBQC2ALcAtAC1AL4AvwEGAQcBCAEJAQoBCwEMAQ0BDgEPARABEQESARMBFAEVARYBFwEYARkBGgEbARwBHQEe'
    'AR8BIAEhASIBIwEkASUBJgEnASgBKQEqASsBLAEtAS4BLwEwATEBMgEzATQBNQE2ATcBOAE5AToBOwE8AT0BPgE/AUABQQFCAUMB'
    'RAFFAUYBRwFIAUkBSgFLAUwBTQFOAU8BUAFRAVIBUwFUAVUBVgFXAVgBWQFaAVsBXAFdAV4BXwFgAWEBYgFjAWQBZQFmAWcBaAFp'
    'AWoBawFsAW0BbgFvAXABcQFyAXMBdAF1AXYBdwF4AXkBegF7AXwBfQF+AX8BgAGBAYIBgwGEAYUBhgGHAYgBiQGKAYsBjAGNAY4B'
    'jwGQAZEBkgGTAZQBlQGWAZcBmAGZAZoBmwGcAZ0BngGfAaABoQGiAaMBpAGlAaYBpwGoAakBqgGrAawBrQGuAa8BsAGxAMMKemVy'
    'b25vam9pbgh6ZXJvam9pbgtsZWZ0dG9yaWdodAtyaWdodHRvbGVmdAV1MDYwQwV1MDYxQgV1MDYxRgV1MDY0MAV1MDY0QgV1MDY0'
    'QwV1MDY0RAV1MDY0RQV1MDY0RgV1MDY1MAV1MDY1MQV1MDY1MgV1RTgyMAV1RTgyMQV1RTgyMgV1RTgyMwV1RTgyNAV1RTgyNQV1'
    'RTgyNgV1RTgyNwV1RTgyOAV1RTgyOQV1RTgyQQV1RTgyQgV1RTgyQwV1RTgyRAV1RkM2MAV1RkM2MQV1RTgxOAV1RkM1RQV1RkM2'
    'MgV1RkM1RgV1MDYyMQV1MDYyMgtIY2lyY3VtZmxleAV1MDYyMwV1RkU4NAV1MDYyNAV1RkU4NgV1MDYyNQV1RkU4OAV1MDYyNgV1'
    'RkU4QQV1RkU4QgV1RkU4QwV1MDYyNwV1RkU4RQV1MDYyOAV1RkU5MAV1RkU5MQV1RkU5MgV1MDY3RQV1RkI1NwV1RkI1OAV1RkI1'
    'OQV1MDYyQQV1RkU5NgV1RkU5NwV1RkU5OAV1MDYyQgV1RkU5QQV1RkU5QgV1RkU5QwV1MDYyQwV1RkU5RQV1RkU5RgV1RkVBMAV1'
    'MDY4NgV1RkI3QgV1RkI3QwV1RkI3RAV1MDYyRAV1RkVBMgV1RkVBMwV1RkVBNAV1MDYyRQV1RkVBNgV1RkVBNwV1RkVBOAV1MDYy'
    'RgV1RkVBQQV1MDYzMAV1RkVBQwV1MDYzMQV1RkVBRQV1MDYzMgV1RkVCMAV1MDY5OAV1RkI4QgV1MDYzMwV1RkVCMgV1RkVCMwV1'
    'RkVCNAV1MDYzNAV1RkVCNgV1RkVCNwV1RkVCOAV1MDYzNQV1RkVCQQV1RkVCQgV1RkVCQwV1MDYzNgV1RkVCRQV1RkVCRgV1RkVD'
    'MAV1MDYzNwV1RkVDMgV1RkVDMwV1RkVDNAV1MDYzOAV1RkVDNgV1RkVDNwV1RkVDOAV1MDYzOQV1RkVDQQV1RkVDQgV1RkVDQwV1'
    'MDYzQQV1RkVDRQV1RkVDRgV1RkVEMAV1MDY0MQV1RkVEMgV1RkVEMwV1RkVENAV1MDY0MgV1RkVENgV1RkVENwV1RkVEOAV1MDY0'
    'MwV1RkVEQQV1RkVEQgV1RkVEQwV1MDZBRgV1RkI5MwV1RkI5NAV1RkI5NQV1MDY0NAV1RkVERQV1RkVERgV1RkVFMAV1MDY0NQV1'
    'RkVFMgV1RkVFMwV1RkVFNAV1MDY0NgV1RkVFNgV1RkVFNwV1RkVFOAV1MDY0OAV1RkVFRQV1MDY0NwV1RkVFQQV1RkVFQgV1RkVF'
    'QwV1MDYyOQV1RkU5NAV1MDY0QQV1RkVGMgV1RkVGMwV1RkVGNAV1RkVGNQV1RkVGNgV1RkVGNwV1RkVGOAV1RkVGOQV1RkVGQQV1'
    'RkVGQgV1RkVGQwV1MDY0OQV1RkVGMAV1RkRGMgAAQCETExISEREQEA8PDg4NDQwMCwsKCgkJCAgHBwYGBQUAAAGNuAH/hUVoREVo'
    'REVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoREVoRLMCAUYAK7MEA0YAK7EBAUVoRLEDA0VoRA=='
)


def _fa_font_css():
    """Self-contained B Nazanin regular/bold and the original OFL fallback."""
    return (
        "@font-face{font-family:'BCD B Nazanin';src:url(data:font/ttf;base64,"
        + BCD_BNAZANIN_REGULAR_TTF_BASE64
        + ") format('truetype');font-weight:400;font-style:normal;font-display:swap;}"
        "@font-face{font-family:'BCD B Nazanin';src:url(data:font/ttf;base64,"
        + BCD_BNAZANIN_BOLD_TTF_BASE64
        + ") format('truetype');font-weight:700;font-style:normal;font-display:swap;}"
        "@font-face{font-family:'Vazirmatn';src:url(data:font/woff2;base64,"
        + WQ_VAZIRMATN_WOFF2_BASE64
        + ") format('woff2');font-weight:100 900;font-style:normal;font-display:swap;}"
        ":root{--bcd-font:'BCD B Nazanin','B Nazanin','BNazanin','Vazirmatn',Tahoma,sans-serif;}"
    )


def _ui_styles():
    st.markdown("<style>" + _fa_font_css() + """
    :root{--bcd-navy:#16324F;--bcd-teal:#087F8C;--bcd-text:#243447;
      --bcd-muted:#526477;--bcd-border:#DCE5ED;--bcd-bg:#F5F7FA;}
    html,body,.stApp{font-family:var(--bcd-font);color:var(--bcd-text);}
    .stApp{background:var(--bcd-bg);color-scheme:light;}
    .block-container{max-width:1520px;padding:2.5rem 2.2rem 3.5rem;}
    [data-testid="stAppViewContainer"]{flex-direction:row-reverse;}
    [data-testid="stSidebar"]{left:auto;right:0;direction:rtl;
      background:#16324F;border-left:1px solid #294760;}
    [data-testid="stSidebar"][aria-expanded="false"]{transform:translateX(100%);}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1rem;}
    [data-testid="stSidebar"] [data-testid="stExpander"]{border-color:#53697F;}
    [data-testid="stSidebarCollapsedControl"]{left:auto;right:1rem;}
    .stMarkdown,.stText,.stCaption,label,h1,h2,h3,h4,h5,h6,summary,
    button,input,textarea,[data-baseweb="select"],[role="tab"],[role="option"],
    [data-testid="stMetric"],[data-testid="stAlert"],
    [data-testid="stCaptionContainer"]{font-family:var(--bcd-font)!important;}
    [data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],[data-testid="stAlert"],
    [data-testid="stMetric"],h1,h2,h3,h4,h5,h6,summary{direction:rtl;text-align:right;}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li{line-height:1.95;}
    [data-testid="stMarkdownContainer"] ul,[data-testid="stMarkdownContainer"] ol{
      padding-inline-start:1.4rem;padding-inline-end:0;}
    h1,h2,h3{color:var(--bcd-navy);line-height:1.7;}
    h2{font-size:1.65rem;}h3{font-size:1.4rem;}
    [data-testid="stCaptionContainer"]{color:var(--bcd-muted);line-height:1.9;}
    [data-testid="stHorizontalBlock"]{direction:rtl;gap:1rem;}
    [data-baseweb="tab-list"]{direction:rtl;gap:1.2rem;border-bottom:1px solid var(--bcd-border);}
    [data-baseweb="tab"]{font-size:1.55rem;font-weight:700;line-height:1.7;
      min-height:3.65rem;padding:.75rem 1rem;color:var(--bcd-muted);}
    [aria-selected="true"][data-baseweb="tab"]{color:var(--bcd-teal);font-weight:700;}
    [data-baseweb="tab-highlight"]{background:var(--bcd-teal);}
    .stTextInput input,.stTextArea textarea,[data-baseweb="select"],
    [role="listbox"],[role="option"],[data-testid="stRadio"]{direction:rtl;text-align:right;}
    .stDateInput input,.stNumberInput input,.stTextInput input{
      direction:ltr;text-align:right;font-variant-numeric:tabular-nums;}
    [data-testid="stMetricValue"]{direction:ltr;unicode-bidi:isolate;text-align:right;}
    [data-testid="stMetric"]{background:white;border:1px solid var(--bcd-border);
      border-radius:12px;padding:1rem 1.2rem;border-top:3px solid var(--bcd-teal);}
    .stButton button,.stDownloadButton button{direction:rtl;min-height:2.9rem;
      border-radius:9px;border:1px solid #B7C8D6;background:white;
      color:var(--bcd-navy);font-weight:700;padding:.5rem 1rem;font-size:1.15rem;}
    .stButton button[kind="primary"],.stButton button[data-testid="stBaseButton-primary"]{
      background:var(--bcd-teal);color:white;border-color:var(--bcd-teal);}
    button:focus-visible,input:focus-visible{outline:3px solid #26A6A1!important;outline-offset:3px;}
    button:disabled{opacity:.55;}
    [data-testid="stAlert"]{border-radius:10px;line-height:1.9;}
    [data-testid="stExpander"]{border:1px solid var(--bcd-border);border-radius:12px;}
    [data-testid="stProgressBar"] [role="progressbar"]{direction:ltr;}
    iframe,.folium-map,canvas,[data-testid="stImage"],
    [data-testid="stPyplot"]{direction:ltr;}
    iframe{border-radius:14px;}
    .bcd-hero{direction:rtl;background:linear-gradient(120deg,#16324F,#164A59);
      padding:1.55rem 1.8rem;border-radius:16px;color:white;margin-bottom:1.2rem;}
    .bcd-hero h1{color:white;font-size:clamp(1.8rem,2.5vw,2.35rem);
      display:flex;align-items:center;gap:.85rem;margin:0 0 .35rem;}
    .bcd-app-icon{display:inline-flex;align-items:center;justify-content:center;
      width:3.1rem;height:3.1rem;flex-shrink:0;border-radius:13px;
      background:rgba(255,255,255,.09);border:1px solid rgba(142,225,216,.3);
      color:#8EE1D8;}
    .bcd-app-icon svg{width:2.15rem;height:2.15rem;}
    .bcd-hero p{color:#D8E9F0;margin:0;line-height:1.9;font-size:1.15rem;}
    .bcd-kicker{color:#8EE1D8;font-size:1rem;letter-spacing:.01em;}
    .bcd-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));direction:rtl;
      gap:.8rem;margin:.8rem 0 1.8rem;}
    .bcd-step{background:white;border:1px solid var(--bcd-border);border-radius:10px;
      display:flex;align-items:center;gap:.65rem;min-width:0;
      padding:1rem 1.05rem;color:var(--bcd-text);font-size:1.35rem;
      font-weight:700;line-height:1.75;}
    .bcd-step b{display:inline-flex;align-items:center;justify-content:center;
      flex-shrink:0;width:2.25rem;height:2.25rem;border-radius:50%;
      color:#075D66;background:#E3F3F3;border:1px solid #CBE3E3;
      font-size:1.4rem;font-weight:700;line-height:1;}
    .bcd-step-label{font-family:var(--bcd-font);font-size:1.35rem;font-weight:700;}
    .bcd-figure-title{direction:rtl;text-align:right;font-weight:650;
      color:var(--bcd-navy);min-height:2.2rem;line-height:1.8;}
    .bcd-roi-card{background:#FFFFFF;border:1px solid var(--bcd-border);
      border-right:3px solid var(--bcd-teal);border-radius:10px;
      padding:.8rem 1rem;line-height:1.9;margin-bottom:.6rem;}
    .bcd-roi-card-title{display:flex;align-items:center;flex-wrap:wrap;gap:.65rem;
      font-size:1.15rem;font-weight:700;margin-bottom:.3rem;}
    .bcd-roi-selected{font-size:.85rem;font-weight:600;line-height:1.9;
      color:#075D66;background:#E3F3F3;border-radius:6px;padding:.1rem .55rem;}
    .bcd-roi-area{font-size:1.05rem;font-weight:500;}
    .bcd-roi-coordinates{color:#526477;font-size:.95rem;line-height:2;
      margin-top:.25rem;overflow-wrap:anywhere;}
    /* Explicit light surfaces and foregrounds also work when the viewer's
       Streamlit preference is dark. Native icon fonts are left untouched. */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"], [data-testid="stText"],
    [data-testid="stRadio"] label, [data-testid="stSpinner"] p{
      color:#243447!important;}
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p,[data-testid="stRadio"] label{
      font-size:1.15rem;line-height:1.9;font-family:var(--bcd-font)!important;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stImageCaption"]{color:#526477!important;font-size:1.05rem;}
    [data-testid="stMetricValue"]{font-size:2rem;font-weight:700;font-family:var(--bcd-font)!important;}
    .st-key-ui_change_result_metrics [data-testid="stMetricLabel"],
    .st-key-ui_change_result_metrics [data-testid="stMetricLabel"] p{
      font-size:1.35rem!important;font-weight:700!important;line-height:1.8;
      white-space:normal!important;overflow:visible;text-overflow:clip;}
    .st-key-ui_change_result_metrics [data-testid="stMetricValue"]{
      font-size:2.25rem!important;letter-spacing:0;}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#16324F!important;}
    [data-testid="stExpander"], [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary{
      background:#FFFFFF!important;color:#243447!important;}
    [data-testid="stExpander"] summary p{
      color:#16324F!important;font-size:1.15rem;}
    [data-testid="stAlert"]{background:#EDF5F8!important;
      color:#16324F!important;border:1px solid #C8DCE5;}
    [data-testid="stAlert"] p,[data-testid="stAlert"] a{
      color:#16324F!important;}
    [data-testid="stWidgetLabel"] label, .stTextInput input, .stTextArea textarea,
    .stDateInput input, .stNumberInput input, [data-baseweb="input"],
    [data-baseweb="base-input"], [data-baseweb="textarea"],
    [data-baseweb="select"] > div{
      background:#FFFFFF!important;color:#243447!important;}
    input,textarea{caret-color:#16324F;-webkit-text-fill-color:#243447;}
    input::placeholder,textarea::placeholder{color:#64748B!important;opacity:1;}
    input:disabled,textarea:disabled{-webkit-text-fill-color:#526477;color:#526477!important;}
    [data-baseweb="popover"], [role="listbox"], [role="option"],
    [data-baseweb="menu"], [data-baseweb="menu"] li,
    [data-baseweb="calendar"], [data-baseweb="calendar"] button{
      background:#FFFFFF!important;color:#243447!important;
      font-family:var(--bcd-font)!important;}
    [role="option"][aria-selected="true"]{background:#E1F2F2!important;color:#075D66!important;}
    [role="option"] div,[role="option"] span,
    [data-baseweb="select"] > div > div{
      color:#243447!important;font-family:var(--bcd-font)!important;}
    [data-baseweb="select"]{color:#243447!important;}
    [data-baseweb="select"] svg{fill:#526477!important;}
    [data-baseweb="tab"] p{color:#526477!important;font-size:1.55rem!important;
      font-weight:700!important;line-height:1.7;}
    [data-baseweb="tab"][aria-selected="true"] p{color:#087F8C!important;}
    .stButton button,.stDownloadButton button{
      background:#FFFFFF!important;color:#16324F!important;}
    .stButton button p,.stDownloadButton button p{
      color:#16324F!important;font-size:1.15rem;}
    .stButton button[kind="primary"],.stButton button[data-testid="stBaseButton-primary"]{
      background:#087F8C!important;border-color:#087F8C!important;color:#FFFFFF!important;}
    .stButton button[kind="primary"] p,
    .stButton button[data-testid="stBaseButton-primary"] p{color:#FFFFFF!important;}
    .stButton button:hover:not(:disabled),.stDownloadButton button:hover:not(:disabled){
      border-color:#087F8C!important;box-shadow:0 0 0 2px rgba(8,127,140,.12);}
    .stButton button:disabled,.stDownloadButton button:disabled{
      background:#E9EFF3!important;color:#526477!important;opacity:1;}
    .stButton button:disabled p,.stDownloadButton button:disabled p{color:#526477!important;}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] > div > div > .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{
      color:#F0F6FA!important;}
    /* Sidebar cards are white surfaces; never inherit the navy area's white text. */
    [data-testid="stSidebar"] [data-testid="stExpander"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] h1,
    [data-testid="stSidebar"] [data-testid="stExpander"] h2,
    [data-testid="stSidebar"] [data-testid="stExpander"] h3,
    [data-testid="stSidebar"] [data-testid="stAlert"] p{
      color:#243447!important;}
    [data-testid="stSidebar"] .stButton button p,
    [data-testid="stSidebar"] .stDownloadButton button p{color:#16324F!important;}
    [data-testid="stSidebar"] .stButton button[kind="primary"] p{
      color:#FFFFFF!important;}
    [data-testid="stSidebar"] .stButton button:disabled p{color:#526477!important;}
    .stApp .bcd-hero h1{color:#FFFFFF!important;}
    .stApp .bcd-hero p{color:#D8E9F0!important;}
    .bcd-roi-card{color:#243447!important;font-size:1.05rem;line-height:1.9;}
    .bcd-roi-card,.bcd-roi-card div,.bcd-roi-card span,.bcd-roi-card bdi{
      font-family:'Vazirmatn',Tahoma,sans-serif!important;}
    .stProgress [role="progressbar"]{direction:ltr;background:#DBE9ED;border-radius:8px;}
    .stProgress [role="progressbar"] > div{background:#087F8C!important;}
    .stProgress p{color:#16324F!important;font-size:1.15rem;}
    .bcd-progress-wait{direction:ltr;height:12px;background:#DBE9ED;
      border-radius:8px;overflow:hidden;margin:.45rem 0;}
    .bcd-progress-wait > span{display:block;height:100%;width:35%;
      border-radius:8px;background:#087F8C;animation:bcd-progress-wait 1.7s ease-in-out infinite;}
    @keyframes bcd-progress-wait{0%{transform:translateX(-100%);}100%{transform:translateX(390%);}}
    /* Streamlit's public container key supplies this CSS class. Hide only
       disposable backend output, independent of surrounding DOM nesting. */
    [class*="st-key-ui_private_output_"]{display:none!important;}
    bdi,code{unicode-bidi:isolate;direction:ltr;}
    hr{border-color:var(--bcd-border);margin:1.6rem 0;}
    @media(max-width:850px){.block-container{padding:2rem 1rem;}
      .bcd-steps{grid-template-columns:repeat(2,minmax(0,1fr));}.bcd-hero{padding:1.15rem;}
      [data-baseweb="tab"],[data-baseweb="tab"] p{font-size:1.35rem!important;}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important;}
      .bcd-progress-wait > span{animation:none;width:100%;opacity:.6;}}
    </style>""", unsafe_allow_html=True)


@contextmanager
def _ui_private_output():
    """Discard technical rendering only; execute every backend statement normally."""
    output = st.empty()
    sequence = st.session_state.get('ui_private_output_sequence', 0)
    st.session_state.ui_private_output_sequence = sequence + 1
    try:
        with output.container():
            with st.container(key=f'ui_private_output_{sequence}'):
                yield
    finally:
        output.empty()


class _UIStageProgress:
    """Forward only existing progress values to the visible UI slot."""

    def __init__(self, presentation):
        self.presentation = presentation

    def progress(self, value, *args, **kwargs):
        self.presentation.show_progress(value)
        return self

    def empty(self):
        self.presentation.waiting()


class _UIProcessingPresentation:
    """Script-local Streamlit view: every non-UI attribute delegates unchanged.

    The shared streamlit module is never monkey-patched. Only this script's
    `st` name is temporarily bound to this view while its unchanged, uncached
    pipeline executes. Session state, caching APIs and all other calls retain
    their original objects and behavior. Progress denominators come exclusively
    from the original pipeline, not estimates of elapsed processing time.
    """

    def __init__(self, original_streamlit, progress_slot, stage_slot):
        self.original = original_streamlit
        self.progress_slot = progress_slot
        self.stage_slot = stage_slot
        self.stage = 'آماده‌سازی پردازش'

    def __getattr__(self, name):
        return getattr(self.original, name)

    def waiting(self):
        self.progress_slot.markdown(
            '<div class="bcd-progress-wait" role="progressbar" '
            'aria-label="پردازش در حال انجام است" '
            'aria-valuetext="در انتظار تکمیل مرحلهٔ جاری"><span></span></div>',
            unsafe_allow_html=True,
        )

    def header(self, body, *args, **kwargs):
        stages = {
            'Phase 1: Cloud-Free Collection': 'مرحلهٔ ۱ از ۴ · آماده‌سازی تصاویر ماهواره‌ای',
            'Phase 2: Download & Gap-Fill': 'مرحلهٔ ۲ از ۴ · دریافت و آماده‌سازی تصاویر',
            'Phase 3: Patch Validity Analysis': 'مرحلهٔ ۳ از ۴ · بررسی کیفیت داده‌ها',
            'Phase 4: Classification': 'مرحلهٔ ۴ از ۴ · شناسایی ساختمان‌ها',
        }
        if body in stages:
            self.stage = stages[body]
            self.stage_slot.markdown('**' + self.stage + '**')
            self.waiting()
        return self.original.header(body, *args, **kwargs)

    def progress(self, value, *args, **kwargs):
        self.show_progress(value)
        return _UIStageProgress(self)

    def show_progress(self, value):
        # Preserve the original numeric progress input exactly. Its Persian
        # label describes the CURRENT stage, not a fabricated overall percent.
        percent = value if isinstance(value, int) else float(value) * 100
        self.progress_slot.progress(
            value, text=self.stage + ' · ' + _roi_number(percent, 0) + '٪'
        )

    def finish(self, success):
        if success:
            self.progress_slot.progress(1.0, text='پردازش با موفقیت انجام شد · ۱۰۰٪')
            self.stage_slot.success('تصاویر و نتایج آمادهٔ مشاهده هستند.')
        else:
            self.progress_slot.empty()
            self.stage_slot.warning('پردازش کامل نشد؛ تنظیمات و وضعیت اتصال را بررسی کنید.')


@contextmanager
def _ui_processing_feedback(progress_slot, stage_slot):
    """Expose native backend progress without changing any processing function."""
    original_streamlit = globals()['st']
    presentation = _UIProcessingPresentation(original_streamlit, progress_slot, stage_slot)
    presentation.waiting()
    stage_slot.markdown('**آماده‌سازی پردازش…**')
    globals()['st'] = presentation
    try:
        yield presentation
    finally:
        globals()['st'] = original_streamlit


def _ui_header():
    st.markdown('''<header class="bcd-hero" dir="rtl">
      <span class="bcd-kicker">سنجش از دور · پایش توسعهٔ شهری</span>
      <h1><span class="bcd-app-icon" aria-hidden="true">
        <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round" focusable="false">
          <path d="M4 28h24M7 28V15h9v13M16 28V10h9v18"/>
          <path d="M10 19h3m-3 4h3m6-9h3m-3 5h3m-3 5h3"/>
          <path d="m5 10 7-6 5 3 10-5m-5 0h5v5"/>
        </svg></span><span>پایش تغییرات ساختمان‌ها</span></h1>
      <p>بررسی تصاویر ماهواره‌ای سنتینل‌ـ۲ در طول زمان و شناسایی ساخت‌وسازهای جدید</p>
      </header>
      <nav class="bcd-steps" aria-label="مراحل تحلیل">
      <div class="bcd-step"><b>۱</b><span class="bcd-step-label">انتخاب محدوده</span></div>
      <div class="bcd-step"><b>۲</b><span class="bcd-step-label">تنظیم بازهٔ زمانی</span></div>
      <div class="bcd-step"><b>۳</b><span class="bcd-step-label">اجرای پردازش</span></div>
      <div class="bcd-step"><b>۴</b><span class="bcd-step-label">مشاهده و دریافت نتایج</span></div></nav>''',
      unsafe_allow_html=True)
    st.sidebar.markdown('### 🛰️ وضعیت سامانه')
    st.sidebar.caption('محدوده و بازهٔ زمانی را انتخاب کنید؛ سپس پردازش را آغاز کنید.')


def _ui_classification_summary():
    config = st.session_state.processing_config or {}
    st.success('پردازش با موفقیت انجام شد. تصاویر آمادهٔ بررسی و دریافت هستند.')
    c1, c2, c3 = st.columns(3)
    c1.metric('🗓️ ماه‌های آماده', len(st.session_state.classification_thumbnails))
    c2.metric('📅 آغاز بازهٔ تحلیل', config.get('start_date', '—'))
    c3.metric('📅 پایان بازه (غیرشامل)', config.get('end_date', '—'))
    st.caption('تصویر رنگ طبیعی و نقشهٔ ساختمان‌ها برای هر ماه نمایش داده می‌شوند. '
               'دکمهٔ زیر هر تصویر، فایل همان ماه را دریافت می‌کند.')


def _ui_comparison_panel(first_rgb_cropped, last_rgb_cropped, change_mask_cropped,
                         first_class_cropped, last_class_cropped, first_month,
                         last_month, stats):
    """Browser-shaped Persian headings; original plot data/colormaps/orientation."""
    titles = [
        f'تصویر رنگ طبیعی آغاز دوره · {first_month}',
        f'تصویر رنگ طبیعی پایان دوره · {last_month}',
        f"ساخت‌وساز جدید · {stats['change_pixels']:,} پیکسل",
        f'ساختمان‌های آغاز دوره · {first_month}',
        f'ساختمان‌های پایان دوره · {last_month}',
        f"نقشهٔ تغییرات · {stats['change_percentage']:.2f}٪",
    ]
    # The same six views as the original figure. Persian text is rendered by
    # the browser, avoiding disconnected Matplotlib glyphs or new RTL packages.
    change_color = np.zeros((*change_mask_cropped.shape, 3), dtype=np.uint8)
    change_color[change_mask_cropped == 1] = [255, 0, 0]
    for row in range(2):
        cols = st.columns(3)
        for col in range(3):
            index = row * 3 + col
            with cols[col]:
                st.markdown('<div class="bcd-figure-title">' + html.escape(titles[index])
                            + '</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                if index == 0:
                    ax.imshow(first_rgb_cropped)
                elif index == 1:
                    ax.imshow(last_rgb_cropped)
                elif index == 2:
                    ax.imshow(last_rgb_cropped)
                    ax.imshow(change_color, alpha=0.6)
                elif index == 3 and first_class_cropped is not None:
                    ax.imshow(first_class_cropped, cmap='Greens')
                elif index == 4 and last_class_cropped is not None:
                    ax.imshow(last_class_cropped, cmap='Reds')
                elif index == 5:
                    ax.imshow(change_mask_cropped, cmap='hot')
                ax.axis('off')
                fig.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close(fig)
    st.caption('در تصاویر مقایسه، قرمز نشان‌دهندهٔ ساخت‌وساز جدید است. '
               'در نقشهٔ تعاملی، لایهٔ تغییرات با آبی نمایش داده می‌شود.')


# Presentation-only ROI controls adapted from the Persian water-quality reference.
# No Earth Engine calls, filesystem persistence or computational cache changes.

def _roi_number(value, decimals=2):
    return f'{value:,.{decimals}f}'.translate(str.maketrans('0123456789,.', '۰۱۲۳۴۵۶۷۸۹٬٫'))


def _roi_area_km2(polygon):
    """Approximate display-only area; never changes the pipeline geometry."""
    if polygon is None:
        return 0.0
    mean_lat = math.radians((polygon.bounds[1] + polygon.bounds[3]) / 2.0)
    return abs(polygon.area) * 110.574 * 111.320 * math.cos(mean_lat)


def _roi_format_area(polygon):
    area = _roi_area_km2(polygon)
    decimals = 0 if area >= 100 else 1 if area >= 10 else 2
    if 0 < area < 0.01:
        return _roi_number(area * 100, 2) + ' هکتار'
    return _roi_number(area, decimals) + ' کیلومتر مربع'


def _roi_map_compat(fmap, **kwargs):
    """Use only arguments supported by the installed streamlit-folium."""
    import inspect
    try:
        parameters = inspect.signature(st_folium).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return st_folium(fmap, **kwargs)


_ROI_INTERACTION_JS = r'''
{% macro script(this, kwargs) %}
(function () {
  var MAP = {{ this._parent.get_name() }};
  if (typeof L === 'undefined' || !MAP) { return; }
  if (L.drawLocal) {
    L.drawLocal.draw.toolbar.actions = {title: 'لغو ترسیم', text: 'لغو'};
    L.drawLocal.draw.toolbar.finish = {title: 'پایان ترسیم', text: 'پایان'};
    L.drawLocal.draw.toolbar.undo = {title: 'حذف آخرین نقطه', text: 'حذف آخرین نقطه'};
    Object.assign(L.drawLocal.draw.toolbar.buttons, {
      polygon: 'رسم چندضلعی', rectangle: 'رسم مستطیل',
      polyline: 'رسم خط', circle: 'رسم دایره', marker: 'نشانه‌گذاری', circlemarker: 'نشانه‌گذاری'
    });
    L.drawLocal.draw.handlers.polygon.tooltip = {
      start: 'برای شروع، روی نقشه کلیک کنید.',
      cont: 'برای افزودن نقطه کلیک کنید.',
      end: 'برای پایان، روی نقطه نخست کلیک کنید.'
    };
    L.drawLocal.draw.handlers.rectangle.tooltip = {start: 'برای رسم مستطیل، کلیک کنید و بکشید.'};
    L.drawLocal.draw.handlers.simpleshape.tooltip = {end: 'برای پایان ترسیم، دکمه ماوس را رها کنید.'};
    L.drawLocal.draw.handlers.polygon.error = 'خطا: ضلع‌ها نباید یکدیگر را قطع کنند.';
  }
  var mapContainer = MAP.getContainer();
  var label = L.DomUtil.create('div', 'roi-live-area', mapContainer);
  label.style.cssText = 'position:absolute;z-index:1000;display:none;pointer-events:none;white-space:nowrap;direction:rtl;transform:translateX(-50%);background:rgba(255,255,255,.97);color:#16324F;border:2px solid #087F8C;border-radius:12px;padding:6px 12px;font-family:var(--bcd-font);font-size:17px;font-weight:700;box-shadow:0 2px 8px rgba(22,50,79,.12)';
  var state = {active:false,type:null,anchor:null,known:null,pts:null};
  function fa(text) {
    return String(text).replace(/[0-9]/g, function (n) {return '۰۱۲۳۴۵۶۷۸۹'[Number(n)];}).replace(/\./g, '٫');
  }
  function geodesicArea(ring) {
    if (!ring || ring.length < 3) {return 0;}
    var sum = 0, d2r = Math.PI / 180;
    for (var i=0; i<ring.length; i++) {
      var p1=ring[i], p2=ring[(i+1)%ring.length];
      sum += (p2.lng-p1.lng)*d2r*(2+Math.sin(p1.lat*d2r)+Math.sin(p2.lat*d2r));
    }
    return Math.abs(sum*6378137*6378137/2);
  }
  function formatArea(m2) {
    var km2=m2/1000000;
    if (km2<0.01) {return fa((m2/10000).toFixed(2))+' هکتار';}
    return fa(km2.toFixed(km2>=100?0:km2>=10?1:2))+' کیلومتر مربع';
  }
  function hide() {label.style.display='none';}
  function reset() {state={active:false,type:null,anchor:null,known:null,pts:null};hide();}
  function draw(ring) {
    var area=geodesicArea(ring);
    if (!isFinite(area)||area<=0) {hide();return;}
    var south=ring[0].lat,west=ring[0].lng,east=ring[0].lng;
    for(var i=1;i<ring.length;i++) {south=Math.min(south,ring[i].lat);west=Math.min(west,ring[i].lng);east=Math.max(east,ring[i].lng);}
    var pt=MAP.latLngToContainerPoint(L.latLng(south,(west+east)/2));
    label.textContent='مساحت تقریبی: '+formatArea(area);
    label.style.left=pt.x+'px';label.style.top=(pt.y+12)+'px';label.style.display='block';state.pts=ring;
  }
  function flatten(latlngs) {
    while(latlngs&&latlngs.length&&latlngs[0]&&latlngs[0].length!==undefined) {latlngs=latlngs[0];}
    return latlngs||[];
  }
  function currentRing(cursor) {
    if(state.type==='rectangle'&&state.anchor&&cursor) {
      var a=state.anchor;
      if(Math.abs(a.lat-cursor.lat)>1e-12&&Math.abs(a.lng-cursor.lng)>1e-12) {
        return [L.latLng(a.lat,a.lng),L.latLng(a.lat,cursor.lng),L.latLng(cursor.lat,cursor.lng),L.latLng(cursor.lat,a.lng)];
      }
      return null;
    }
    var found=null;
    MAP.eachLayer(function(layer) {
      if(!found&&layer&&typeof layer.getLatLngs==='function'&&!(state.known&&state.known[L.Util.stamp(layer)])) {found=layer;}
    });
    if(!found) {return null;}
    var ring=flatten(found.getLatLngs()).slice();
    if(!(L.Rectangle&&found instanceof L.Rectangle)&&cursor) {ring.push(cursor);}
    return ring;
  }
  function move(cursor) {
    if(!state.active||!cursor) {return;}
    try {draw(currentRing(cursor));} catch(ignore) {hide();}
  }
  MAP.on('draw:drawstart',function(e) {
    state={active:true,type:e.layerType,anchor:null,known:{},pts:null};
    MAP.eachLayer(function(layer) {state.known[L.Util.stamp(layer)]=true;});hide();
  });
  MAP.on('draw:drawstop draw:created draw:canceled',reset);
  MAP.on('mousedown',function(e) {if(state.active&&!state.anchor&&e.latlng) {state.anchor=e.latlng;}});
  MAP.on('mousemove',function(e) {if(e.latlng) {move(e.latlng);}});
  mapContainer.addEventListener('mousedown',function(ev) {
    if(state.active&&!state.anchor) {try {state.anchor=MAP.mouseEventToLatLng(ev);} catch(ignore) {}}
  },true);
  mapContainer.addEventListener('mousemove',function(ev) {
    if(state.active) {try {move(MAP.mouseEventToLatLng(ev));} catch(ignore) {}}
  },true);
  MAP.on('move zoom',function() {if(state.active&&state.pts) {draw(state.pts);}});
  setTimeout(function() {
    var zoomIn=mapContainer.querySelector('.leaflet-control-zoom-in');
    var zoomOut=mapContainer.querySelector('.leaflet-control-zoom-out');
    var layers=mapContainer.querySelector('.leaflet-control-layers-toggle');
    var exportControl=document.getElementById('export');
    if(zoomIn) {zoomIn.title='بزرگ‌نمایی';zoomIn.setAttribute('aria-label','بزرگ‌نمایی');}
    if(zoomOut) {zoomOut.title='کوچک‌نمایی';zoomOut.setAttribute('aria-label','کوچک‌نمایی');}
    if(layers) {layers.title='انتخاب لایه نقشه';layers.setAttribute('aria-label','انتخاب لایه نقشه');}
    if(exportControl) {exportControl.textContent='دریافت محدوده';exportControl.title='دریافت محدوده‌های نقشه با قالب GeoJSON';}
  },0);
})();
{% endmacro %}
'''


def _roi_build_map(busy, selected_index):
    from branca.element import MacroElement
    from jinja2 import Template

    class _RoiInteraction(MacroElement):
        _template = Template(_ROI_INTERACTION_JS)

        def __init__(self):
            super().__init__()
            self._name = 'RoiInteraction'

    saved = st.session_state.drawn_polygons
    draft = st.session_state.last_drawn_polygon
    draft_is_saved = draft is not None and any(p.equals(draft) for p in saved)
    shapes = list(saved) + ([draft] if draft is not None and not draft_is_saved else [])
    if shapes:
        west = min(p.bounds[0] for p in shapes)
        south = min(p.bounds[1] for p in shapes)
        east = max(p.bounds[2] for p in shapes)
        north = max(p.bounds[3] for p in shapes)
        center = [(south + north) / 2, (west + east) / 2]
    else:
        center = [35.6892, 51.3890]
    fmap = folium.Map(location=center, zoom_start=8, control_scale=True, tiles=None)
    folium.TileLayer('OpenStreetMap', name='نقشه پایه').add_to(fmap)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='تصویر ماهواره‌ای',
    ).add_to(fmap)
    # The map itself remains LTR; only text overlays and controls use RTL.
    fmap.get_root().header.add_child(folium.Element(
        '<style>' + _fa_font_css() + '.leaflet-container{direction:ltr;font-family:var(--bcd-font);}'
        '.leaflet-tooltip,.leaflet-popup-content,.leaflet-control-layers,.leaflet-draw-tooltip{direction:rtl;text-align:right;font-family:var(--bcd-font);font-size:17px;}'
        '.leaflet-draw-actions{font-family:var(--bcd-font);}'
        '#export{top:auto!important;bottom:30px;right:10px;font-family:var(--bcd-font)!important;font-size:17px!important;}</style>'
    ))
    # Add localization before Draw is initialized so button labels are Persian.
    fmap.add_child(_RoiInteraction())
    layers = folium.FeatureGroup(name='محدوده‌های مورد مطالعه', show=True)
    for i, polygon in enumerate(saved):
        selected = i == selected_index
        color = '#B36B15' if selected else '#087F8C'
        label = 'محدوده ' + _roi_number(i + 1, 0)
        if selected:
            label += '؛ انتخاب‌شده برای پردازش'
        folium.Polygon(
            locations=[(y, x) for x, y in polygon.exterior.coords],
            color=color, weight=4 if selected else 3, fill=True,
            fill_color=color, fill_opacity=0.19 if selected else 0.10,
            tooltip=label,
        ).add_to(layers)
    if draft is not None and not draft_is_saved:
        folium.Polygon(
            locations=[(y, x) for x, y in draft.exterior.coords],
            color='#C44747', weight=3, dash_array='8,6', fill=True,
            fill_color='#C44747', fill_opacity=0.10,
            tooltip='محدوده رسم‌شده؛ ذخیره‌نشده',
        ).add_to(layers)
    layers.add_to(fmap)
    if not busy:
        # Share the permanent ROI group so the original GeoJSON export keeps
        # working after Streamlit remounts the drawing widget.
        plugins.Draw(
            export=True, feature_group=layers, position='topright',
            show_geometry_on_click=False,
            draw_options={
                'polyline': False, 'rectangle': {'shapeOptions': {'color': '#087F8C'}},
                'polygon': {'allowIntersection': False, 'shapeOptions': {'color': '#087F8C'}},
                'circle': False, 'marker': False, 'circlemarker': False,
            },
            edit_options={'edit': False, 'remove': False},
        ).add_to(fmap)
    folium.LayerControl(position='topleft', collapsed=True).add_to(fmap)
    if shapes:
        fmap.fit_bounds([[south, west], [north, east]], padding=(30, 30))
    return fmap


def render_roi_selection():
    """Render editable ROI UI and return the original selected Polygon."""
    import html

    st.session_state.setdefault('ui_roi_map_version', 0)
    busy = st.session_state.processing_in_progress
    saved = st.session_state.drawn_polygons
    if saved:
        st.session_state.selected_region_index = min(max(0, st.session_state.selected_region_index), len(saved) - 1)
    else:
        st.session_state.selected_region_index = 0
    selected_index = st.session_state.selected_region_index
    st.caption('با ابزار مستطیل یا چندضلعی، محدوده را رسم کنید. مساحت تقریبی هنگام ترسیم روی نقشه نمایش داده می‌شود.')
    version = st.session_state.ui_roi_map_version
    map_data = _roi_map_compat(
        _roi_build_map(busy, selected_index),
        key=f'ui_roi_{"locked" if busy else "map"}_{version}',
        width=1400, height=650, use_container_width=True,
        returned_objects=[] if busy else ['last_active_drawing'],
    )
    if busy:
        st.caption('🔒 نقشه در زمان پردازش فقط برای مشاهده است.')
    if not busy and map_data and map_data.get('last_active_drawing'):
        geometry = (map_data['last_active_drawing'].get('geometry') or {})
        if geometry.get('type') == 'Polygon':
            try:
                candidate = Polygon(geometry['coordinates'][0])
                if candidate.is_empty or not candidate.is_valid or candidate.area <= 0:
                    st.warning('محدوده رسم‌شده معتبر نیست؛ لطفاً یک محدوده بدون تقاطع رسم کنید.')
                elif st.session_state.last_drawn_polygon is None or not st.session_state.last_drawn_polygon.equals(candidate):
                    # Preserve exact exterior coordinates and remount the widget
                    # to discard the temporary browser drawing after capture.
                    st.session_state.last_drawn_polygon = candidate
                    st.session_state.ui_roi_map_version += 1
                    st.rerun()
            except (KeyError, TypeError, ValueError, IndexError):
                st.warning('محدوده قابل خواندن نیست؛ لطفاً دوباره آن را رسم کنید.')
    draft = st.session_state.last_drawn_polygon
    has_draft = draft is not None and not any(p.equals(draft) for p in saved)
    if has_draft:
        st.info('مساحت محدودهٔ رسم‌شده: ' + _roi_format_area(draft))
    if has_draft and saved:
        st.caption('برای پردازش محدودهٔ جدید، ابتدا آن را ذخیره کنید. تا آن زمان، محدودهٔ انتخاب‌شده از فهرست پردازش می‌شود.')
    save_col, discard_col = st.columns([3, 1])
    if save_col.button('💾 ذخیره محدوده', key='ui_roi_save', type='primary',
                       use_container_width=True, disabled=busy or draft is None or not has_draft):
        if not any(p.equals(draft) for p in saved):
            saved.append(draft)
            st.session_state.selected_region_index = len(saved) - 1
            st.session_state.ui_roi_map_version += 1
            st.session_state.ui_roi_notice = 'محدوده ذخیره شد.'
            st.rerun()
    if discard_col.button('✕ حذف محدوده رسم‌شده', key='ui_roi_discard',
                          use_container_width=True, disabled=busy or not has_draft):
        st.session_state.last_drawn_polygon = None
        st.session_state.ui_roi_map_version += 1
        st.session_state.ui_roi_notice = 'محدوده رسم‌شده از نقشه حذف شد.'
        st.rerun()
    notice = st.session_state.pop('ui_roi_notice', None)
    if notice:
        st.success(notice)
    if saved:
        st.markdown('#### 📍 محدوده‌های ذخیره‌شده')
        for i, polygon in enumerate(list(saved)):
            info_col, delete_col = st.columns([5, 1])
            selected_badge = ('<span class="bcd-roi-selected">انتخاب‌شده</span>'
                              if i == st.session_state.selected_region_index else '')
            info_col.markdown(
                '<div class="bcd-roi-card" dir="rtl"><div class="bcd-roi-card-title">محدوده '
                + _roi_number(i + 1, 0) + selected_badge
                + '</div><div class="bcd-roi-area">مساحت: '
                + html.escape(_roi_format_area(polygon))
                + '</div><div class="bcd-roi-coordinates">مرکز محدوده (عرض، طول جغرافیایی): '
                + '<bdi dir="ltr">' + f'{polygon.centroid.y:.4f}, {polygon.centroid.x:.4f}'
                + '</bdi></div></div>',
                unsafe_allow_html=True,
            )
            if delete_col.button('🗑 حذف محدوده', key=f'ui_roi_delete_{version}_{i}',
                                 use_container_width=True, disabled=busy):
                removed = saved.pop(i)
                if draft is not None and draft.equals(removed):
                    st.session_state.last_drawn_polygon = None
                previous_index = st.session_state.selected_region_index
                if i < previous_index:
                    previous_index -= 1
                st.session_state.selected_region_index = min(previous_index, max(0, len(saved) - 1))
                st.session_state.ui_roi_map_version += 1
                st.session_state.ui_roi_notice = 'محدوده از فهرست و نقشه حذف شد.'
                st.rerun()
        selected = st.selectbox(
            '🎯 محدوده مورد استفاده در پردازش', range(len(saved)),
            index=st.session_state.selected_region_index,
            format_func=lambda i: 'محدوده ' + _roi_number(i + 1, 0),
            key=f'ui_roi_selector_{version}', disabled=busy,
        )
        if selected != st.session_state.selected_region_index:
            st.session_state.selected_region_index = selected
            st.session_state.ui_roi_map_version += 1
            st.rerun()
        selected_polygon = saved[selected]
    else:
        selected_polygon = draft
        if draft is not None:
            st.caption('محدوده رسم‌شده برای پردازش انتخاب شده است؛ برای نگه‌داشتن آن در فهرست، «ذخیره محدوده» را بزنید.')
    if selected_polygon is None:
        st.info('برای شروع، یک محدوده روی نقشه رسم کنید.')
    return selected_polygon


def display_thumbnails(thumbnails, valid_months=None):
    if not thumbnails:
        return
    
    mode = st.radio("شیوهٔ نمایش تصاویر", ["Side by Side", "Classification", "RGB"],
                    format_func=lambda value: {"Side by Side": "مقایسهٔ کنار هم",
                        "Classification": "نقشهٔ ساختمان‌ها", "RGB": "تصویر رنگ طبیعی"}[value],
                    horizontal=True)
    st.divider()
    
    if mode == "Side by Side":
        for i in range(0, len(thumbnails), 2):
            cols = st.columns(4)
            for j in range(2):
                idx = i + j
                if idx < len(thumbnails):
                    t = thumbnails[idx]
                    pct = (t['building_pixels'] / t['total_pixels']) * 100
                    suffix = " (تکمیل‌شده)" if t.get('gap_filled') else ""
                    cols[j*2].image(t['rgb_image'], caption=f"تصویر رنگ طبیعی · {t['month_name']}{suffix}")
                    # Add download button under RGB
                    if valid_months and t['month_name'] in valid_months:
                        image_path = valid_months[t['month_name']]
                        with _ui_private_output():
                            image_data = get_image_download_data(image_path, t['month_name'])
                        if image_data is None:
                            st.warning('فایل تصویر این ماه برای دریافت در دسترس نیست.')
                        if image_data:
                            cols[j*2].download_button(
                                label=f"⬇️ دریافت تصویر {t['month_name']}",
                                data=image_data,
                                file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                mime="image/tiff",
                                key=f"dl_sidebyside_{t['month_name']}"
                            )
                    cols[j*2+1].image(t['classification_image'], caption=f"ساختمان‌ها · {t['month_name']} · {pct:.1f}٪")
                    # Add download button under classification image (georeferenced, WGS84/EPSG:4326)
                    if valid_months and t['month_name'] in valid_months and t['month_name'] in st.session_state.probability_maps:
                        source_path = valid_months[t['month_name']]
                        prob_map = st.session_state.probability_maps[t['month_name']]
                        with _ui_private_output():
                            class_geotiff_data = get_classification_geotiff_wgs84(prob_map, source_path, t['month_name'])
                        if class_geotiff_data is None:
                            st.warning('آماده‌سازی فایل مکانی ساختمان‌های این ماه انجام نشد.')
                        if class_geotiff_data:
                            cols[j*2+1].download_button(
                                label=f"⬇️ دریافت تصویر {t['month_name']}",
                                data=class_geotiff_data,
                                file_name=f"classified_{t['month_name']}_wgs84.tif",
                                mime="image/tiff",
                                key=f"dl_class_sidebyside_{t['month_name']}"
                            )
    else:
        key = 'classification_image' if mode == "Classification" else 'rgb_image'
        for row in range((len(thumbnails) + 3) // 4):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c
                if idx < len(thumbnails):
                    t = thumbnails[idx]
                    pct = (t['building_pixels'] / t['total_pixels']) * 100
                    cap = f"ساختمان‌ها · {t['month_name']} · {pct:.1f}٪" if mode == "Classification" else f"تصویر رنگ طبیعی · {t['month_name']}"
                    if t.get(key):
                        cols[c].image(t[key], caption=cap)
                        # Add download button under RGB images only
                        if mode == "RGB" and valid_months and t['month_name'] in valid_months:
                            image_path = valid_months[t['month_name']]
                            with _ui_private_output():
                                image_data = get_image_download_data(image_path, t['month_name'])
                            if image_data is None:
                                st.warning('فایل تصویر این ماه برای دریافت در دسترس نیست.')
                            if image_data:
                                cols[c].download_button(
                                    label="⬇️ دریافت تصویر",
                                    data=image_data,
                                    file_name=f"sentinel2_{t['month_name']}_12bands.tif",
                                    mime="image/tiff",
                                    key=f"dl_rgb_{t['month_name']}"
                                )
                        # Add download button under classification images only (georeferenced, WGS84/EPSG:4326)
                        if mode == "Classification" and valid_months and t['month_name'] in valid_months and t['month_name'] in st.session_state.probability_maps:
                            source_path = valid_months[t['month_name']]
                            prob_map = st.session_state.probability_maps[t['month_name']]
                            with _ui_private_output():
                                class_geotiff_data = get_classification_geotiff_wgs84(prob_map, source_path, t['month_name'])
                            if class_geotiff_data is None:
                                st.warning('آماده‌سازی فایل مکانی ساختمان‌های این ماه انجام نشد.')
                            if class_geotiff_data:
                                cols[c].download_button(
                                    label="⬇️ دریافت تصویر",
                                    data=class_geotiff_data,
                                    file_name=f"classified_{t['month_name']}_wgs84.tif",
                                    mime="image/tiff",
                                    key=f"dl_class_grid_{t['month_name']}"
                                )


# =============================================================================
# Main Application
# =============================================================================
def main():
    _ui_styles()
    _ui_header()
    tab1, tab2 = st.tabs(["🗺️ انتخاب محدوده و پردازش", "📊 شناسایی تغییرات و نتایج"])
    with tab1:
        main_classification_tab()
    with tab2:
        change_detection_tab()

def main_classification_tab():
       
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error("اتصال به سرویس تصاویر ماهواره‌ای برقرار نشد. تنظیمات موجود اتصال را بررسی کنید و دوباره تلاش کنید.")
        st.stop()
    st.sidebar.success("✓ سرویس تصاویر ماهواره‌ای آماده است")
    
    # Model
    model_path = "best_model_version_Unet++_v02_e8.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("در حال آماده‌سازی مدل؛ لطفاً منتظر بمانید…"):
            with _ui_private_output():
                downloaded_model = download_model_from_gdrive("", model_path)
        if not downloaded_model:
            st.error("دریافت مدل انجام نشد. اتصال شبکه و دسترسی به فایل مدل را بررسی کنید.")
            st.stop()
    
    if not st.session_state.model_loaded:
        with st.spinner("در حال بارگذاری مدل…"):
            with _ui_private_output():
                model, device = load_model(model_path)
        if model:
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.sidebar.success("✓ مدل تحلیل آماده است")
        else:
            st.error("بارگذاری مدل انجام نشد. سازگاری فایل مدل و محیط اجرای اصلی را بررسی کنید.")
            st.stop()
    else:
        st.sidebar.success("✓ مدل تحلیل آماده است")

    nodata_pct = 0
    
    with st.sidebar.expander("🗂️ مدیریت داده‌های پردازش", expanded=False):
        st.caption("پاک‌کردن داده‌ها، نتایج و اطلاعات ادامهٔ پردازش جاری را از نشست حذف می‌کند.")
        # Cache Status
        st.header("🗂️ وضعیت ادامهٔ پردازش")
        
        cache_info = []
        if st.session_state.month_analysis_results:
            cache_info.append(f"📊 {len(st.session_state.month_analysis_results)} ماه بررسی شده")
        if st.session_state.downloaded_images:
            cache_info.append(f"📥 {len(st.session_state.downloaded_images)} تصویر دریافت شده")
        if st.session_state.processed_months:
            cache_info.append(f"🧠 {len(st.session_state.processed_months)} ماه پردازش شده")
        
        if cache_info:
            for info in cache_info:
                st.success(info)
        else:
            st.info("هنوز پردازشی ذخیره نشده است")
        
        if st.session_state.failed_downloads:
            st.warning(f"ماه‌های نیازمند بررسی دوباره: {', '.join(st.session_state.failed_downloads)}")
        
        # Processing status indicator
        if st.session_state.processing_in_progress:
            st.error("⏳ پردازش در حال انجام است")
        
        if st.button("🗑️ پاک‌کردن داده‌های پردازش", disabled=st.session_state.processing_in_progress):
            for key in ['processed_months', 'downloaded_images', 'classification_thumbnails', 
                        'valid_patches_mask', 'valid_months', 'current_temp_dir', 'month_analysis_results',
                        'failed_downloads', 'analysis_complete', 'download_complete',
                        'processing_params', 'processing_config', 'pdf_report',
                        'probability_maps', 'change_detection_result', 'change_timing_map']:
                if key in st.session_state:
                    if isinstance(st.session_state[key], dict):
                        st.session_state[key] = {}
                    elif isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.session_state.processing_complete = False
            st.session_state.processing_in_progress = False
            st.rerun()

    # Stop processing button
    if st.session_state.processing_in_progress:
        if st.sidebar.button("⏹️ توقف پردازش", type="primary"):
            st.session_state.processing_in_progress = False
            st.session_state.processing_config = None
            st.warning("پردازش متوقف شد؛ امکان ادامه در فرصتی دیگر وجود دارد.")
            st.rerun()
    
    st.header("۱ · انتخاب محدودهٔ مورد مطالعه")
    selected_polygon = render_roi_selection()

    # Date
    st.header("۲ · تنظیم بازهٔ زمانی")
    c1, c2 = st.columns(2)
    start = c1.date_input("تاریخ آغاز (شامل این تاریخ)", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("تاریخ پایان (غیرشامل)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress,
                        help="تاریخ‌ها میلادی هستند. ماه تاریخ پایان پردازش نمی‌شود؛ مثلاً با پایان 2024-06-01، آخرین ماه پردازش مه ۲۰۲۴ است.")
    
    if start >= end:
        st.error("تاریخ پایان باید پس از تاریخ آغاز باشد.")
        st.stop()
    
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if months < 1:
        st.error("برای تحلیل ماهانه، تاریخ پایان را دست‌کم در ماه بعد از تاریخ آغاز انتخاب کنید.")
        st.stop()
    
    # Calculate first and last month names
    first_month = f"{start.year}-{start.month:02d}"
    last_year = start.year + (start.month - 1 + months - 1) // 12
    last_month_num = (start.month - 1 + months - 1) % 12 + 1
    last_month = f"{last_year}-{last_month_num:02d}"
    
    st.info(f"📅 **{months} ماه**؛ از **{first_month}** تا **{last_month}**. تاریخ پایان {end.strftime('%Y-%m-%d')} در بازه لحاظ نمی‌شود.")
    
    # Process
    st.header("۳ · اجرای پردازش")
    
    # Processing buttons
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_new = st.button(
            "▶ شروع پردازش", 
            type="primary", 
            disabled=st.session_state.processing_in_progress or selected_polygon is None
        )
    
    with col2:
        has_cache = bool(st.session_state.downloaded_images or st.session_state.month_analysis_results)
        resume_btn = st.button(
            "ادامهٔ پردازش قبلی", 
            disabled=not has_cache or st.session_state.processing_in_progress or (selected_polygon is None and st.session_state.processing_config is None)
        )
    
    with col3:
        has_failed = bool(st.session_state.failed_downloads)
        retry_btn = st.button(
            "تلاش دوباره برای موارد ناموفق", 
            disabled=not has_failed or st.session_state.processing_in_progress or (selected_polygon is None and st.session_state.processing_config is None)
        )
    
    # Determine if we should process
    should_process = False
    resume_mode = False
    
    if start_new:
        should_process = True
        resume_mode = False
        # Clear cache for fresh start
        st.session_state.month_analysis_results = {}
        st.session_state.analysis_complete = False
        st.session_state.download_complete = False
        st.session_state.processed_months = {}
        st.session_state.classification_thumbnails = []
        st.session_state.valid_patches_mask = None
        st.session_state.valid_months = {}
        st.session_state.pdf_report = None
        st.session_state.failed_downloads = []
        st.session_state.processing_complete = False
        st.session_state.probability_maps = {}
        st.session_state.change_detection_result = None
        
        # Store processing config
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct,
            'nodata_pct': nodata_pct
        }
        st.session_state.processing_in_progress = True
    
    elif resume_btn:
        should_process = True
        resume_mode = True
        st.session_state.processing_in_progress = True
        
        # Use stored config or create new
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
                'nodata_pct': nodata_pct
            }
    
    elif retry_btn:
        should_process = True
        resume_mode = True
        st.session_state.failed_downloads = []
        st.session_state.processing_in_progress = True
        
        # Use stored config or create new
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
                'nodata_pct': nodata_pct
            }
    
    # Auto-continue if processing was in progress (handles page reruns)
    elif st.session_state.processing_in_progress and st.session_state.processing_config is not None:
        should_process = True
        resume_mode = True
        st.info("در حال ادامهٔ پردازش قبلی…")
    
    # Execute processing
    if should_process:
        config = st.session_state.processing_config
        
        if config is None:
            st.error("تنظیمات پردازش در دسترس نیست؛ محدوده و بازهٔ زمانی را دوباره انتخاب کنید.")
            st.session_state.processing_in_progress = False
            st.stop()
        
        # Create polygon from stored coords
        poly = Polygon(config['polygon_coords'])
        aoi = ee.Geometry.Polygon([config['polygon_coords']])
        
        try:
            processing_status = st.status("در حال دریافت تصاویر و تحلیل ساختمان‌ها…", expanded=False)
            progress_slot = st.empty()
            stage_slot = st.empty()
            st.caption("نوار، پیشرفت مرحلهٔ جاری را نشان می‌دهد؛ با شروع مرحلهٔ بعد از نو محاسبه می‌شود.")
            with st.spinner("پردازش در حال انجام است؛ زمان اجرا به محدوده و تعداد ماه‌ها بستگی دارد."):
                with _ui_processing_feedback(progress_slot, stage_slot) as feedback:
                    with _ui_private_output():
                        thumbs = process_timeseries(
                            aoi, 
                            config['start_date'], 
                            config['end_date'],
                            st.session_state.model, 
                            st.session_state.device,
                            config['cloudy_pct'], 
                            10, 
                            config['nodata_pct'], 
                            resume=resume_mode
                        )
            feedback.finish(bool(thumbs))
            if thumbs:
                processing_status.update(label="پردازش با موفقیت انجام شد", state="complete")
            else:
                processing_status.update(label="پردازش بدون نتیجه پایان یافت", state="error")
                st.error("تصویر قابل استفاده‌ای تولید نشد. دسترسی به تصاویر و بازهٔ زمانی را بررسی کنید؛ امکان ادامهٔ پردازش قبلی وجود دارد.")
            
            if thumbs:
                st.session_state.classification_thumbnails = thumbs
                st.session_state.processing_complete = True
            
            # Processing finished
            st.session_state.processing_in_progress = False
            
        except Exception as e:
            st.error("پردازش کامل نشد. اتصال شبکه و دسترسی به داده‌ها را بررسی کنید؛ سپس از گزینهٔ ادامهٔ پردازش استفاده کنید.")
            if 'progress_slot' in locals():
                progress_slot.empty()
            if 'stage_slot' in locals():
                stage_slot.warning("پردازش متوقف شد؛ امکان ادامه یا تلاش دوباره وجود دارد.")
            processing_status.update(label="پردازش کامل نشد؛ امکان تلاش دوباره وجود دارد", state="error")
            st.session_state.processing_in_progress = False
    
    # Display Results
    if st.session_state.processing_complete and st.session_state.classification_thumbnails:
        st.divider()
        st.header("۴ · تصاویر و نتایج پردازش")
        _ui_classification_summary()
        display_thumbnails(
            st.session_state.classification_thumbnails,
            valid_months=st.session_state.valid_months
        )
        
        # Info about change detection tab
        if st.session_state.probability_maps:
            st.success(f"{len(st.session_state.probability_maps)} ماه برای بررسی تغییرات آماده است. در زبانهٔ «شناسایی تغییرات و نتایج»، تنظیمات را بررسی کنید و تحلیل تغییرات را اجرا کنید.")


def display_interactive_map(first_image_path, last_image_path, first_class, last_class, 
                            change_mask, first_month, last_month, crop_bounds, original_size):
    """Display interactive Folium map with all layers - matching old app style"""
    import folium
    from folium import plugins
    import streamlit.components.v1 as components
    import tempfile
    import base64
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    # Extract years from month names for legend
    first_year = first_month.split('-')[0] if first_month else "N/A"
    last_year = last_month.split('-')[0] if last_month else "N/A"
    
    st.info("""
    لایه‌های نقشه را از پنل بالای سمت راست روشن یا خاموش کنید. برای مشاهده بزرگ‌تر،
    دکمه تمام‌صفحه در سمت چپ نقشه را بزنید.
    رنگ سبز نشان‌دهنده ساختمان‌های ابتدای دوره، قرمز ساختمان‌های پایان دوره و آبی تغییرات شناسایی‌شده است.
    نمایش نقشه پایه به اینترنت نیاز دارد؛ لایه‌های نتایج در نقشه تعبیه شده‌اند.
    """)
    
    try:
        # Get georeferencing info from first image
        if not first_image_path or not os.path.exists(first_image_path):
            st.warning("نمایش نقشه ممکن نیست؛ تصویر مرجع در دسترس نیست.")
            return
        
        with rasterio.open(first_image_path) as src:
            utm_crs = src.crs
            utm_transform = src.transform
            utm_bounds = src.bounds
            utm_height = src.height
            utm_width = src.width
            
            # Calculate center for map
            center_x = (utm_bounds.left + utm_bounds.right) / 2
            center_y = (utm_bounds.bottom + utm_bounds.top) / 2
        
        # Reproject center to WGS84
        from rasterio.warp import transform as rio_transform
        center_lon, center_lat = rio_transform(utm_crs, 'EPSG:4326', [center_x], [center_y])
        center = [center_lat[0], center_lon[0]]
        
        temp_dir = tempfile.mkdtemp()
        dst_crs = 'EPSG:4326'
        
        # Variables to store reprojected paths and target transform
        target_transform = None
        target_width = None
        target_height = None
        target_bounds = None
        
        before_class_wgs84_path = None
        after_class_wgs84_path = None
        change_mask_wgs84_path = None
        before_rgb_wgs84_path = None
        after_rgb_wgs84_path = None
        
        # Helper function to reproject single band data
        def reproject_to_wgs84(data, data_name, dtype='uint8'):
            nonlocal target_transform, target_width, target_height, target_bounds
            
            utm_path = os.path.join(temp_dir, f"{data_name}_utm.tif")
            wgs84_path = os.path.join(temp_dir, f"{data_name}_wgs84.tif")
            
            # Save as UTM GeoTIFF
            with rasterio.open(
                utm_path, 'w', driver='GTiff',
                height=data.shape[0], width=data.shape[1],
                count=1, dtype=dtype, crs=utm_crs, transform=utm_transform
            ) as dst:
                dst.write(data.astype(dtype), 1)
            
            # Reproject to WGS84
            with rasterio.open(utm_path) as src_utm:
                if target_transform is None:
                    # Calculate target transform once
                    target_transform, target_width, target_height = calculate_default_transform(
                        src_utm.crs, dst_crs, src_utm.width, src_utm.height, *src_utm.bounds
                    )
                
                dst_kwargs = src_utm.meta.copy()
                dst_kwargs.update({
                    'crs': dst_crs,
                    'transform': target_transform,
                    'width': target_width,
                    'height': target_height
                })
                
                with rasterio.open(wgs84_path, 'w', **dst_kwargs) as dst_wgs:
                    reproject(
                        source=rasterio.band(src_utm, 1),
                        destination=rasterio.band(dst_wgs, 1),
                        src_transform=src_utm.transform,
                        src_crs=src_utm.crs,
                        dst_transform=target_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
                    if target_bounds is None:
                        target_bounds = dst_wgs.bounds
            
            return wgs84_path
        
        # Helper function to reproject RGB data
        def reproject_rgb_to_wgs84(rgb_data, data_name):
            nonlocal target_transform, target_width, target_height, target_bounds
            
            utm_path = os.path.join(temp_dir, f"{data_name}_utm.tif")
            wgs84_path = os.path.join(temp_dir, f"{data_name}_wgs84.tif")
            
            # Save as UTM GeoTIFF (3 bands)
            with rasterio.open(
                utm_path, 'w', driver='GTiff',
                height=rgb_data.shape[0], width=rgb_data.shape[1],
                count=3, dtype='uint8', crs=utm_crs, transform=utm_transform
            ) as dst:
                for i in range(3):
                    dst.write(rgb_data[:, :, i], i + 1)
            
            # Reproject to WGS84
            with rasterio.open(utm_path) as src_utm:
                if target_transform is None:
                    target_transform, target_width, target_height = calculate_default_transform(
                        src_utm.crs, dst_crs, src_utm.width, src_utm.height, *src_utm.bounds
                    )
                
                dst_kwargs = src_utm.meta.copy()
                dst_kwargs.update({
                    'crs': dst_crs,
                    'transform': target_transform,
                    'width': target_width,
                    'height': target_height
                })
                
                with rasterio.open(wgs84_path, 'w', **dst_kwargs) as dst_wgs:
                    for i in range(1, 4):
                        reproject(
                            source=rasterio.band(src_utm, i),
                            destination=rasterio.band(dst_wgs, i),
                            src_transform=src_utm.transform,
                            src_crs=src_utm.crs,
                            dst_transform=target_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )
                    if target_bounds is None:
                        target_bounds = dst_wgs.bounds
            
            return wgs84_path
        
        # Helper function to convert raster to Folium overlay
        def raster_to_folium_overlay(raster_path, colormap='viridis', opacity=0.7, 
                                     is_binary=False, is_change_mask=False, is_rgb=False):
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                bounds_latlon = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
                
                if is_rgb and src.count >= 3:
                    # RGB image
                    rgb_data = src.read([1, 2, 3])
                    img_array = np.transpose(rgb_data, (1, 2, 0))
                    pil_img = Image.fromarray(img_array.astype(np.uint8))
                elif is_binary:
                    # Binary classification mask
                    data = src.read(1)
                    rgba_array = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
                    mask_val = data > 0
                    if colormap == 'Greens':
                        rgba_array[mask_val, 0:3] = [0, 255, 0]  # Green
                    elif colormap == 'Reds':
                        rgba_array[mask_val, 0:3] = [255, 0, 0]  # Red
                    rgba_array[mask_val, 3] = 180  # Alpha
                    pil_img = Image.fromarray(rgba_array, 'RGBA')
                elif is_change_mask:
                    # Change detection mask - DARK BLUE for better contrast
                    data = src.read(1)
                    rgba_array = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
                    mask_val = data > 0
                    rgba_array[mask_val, 0:3] = [0, 0, 180]  # Dark blue RGB
                    rgba_array[mask_val, 3] = 220  # Higher opacity for visibility
                    pil_img = Image.fromarray(rgba_array, 'RGBA')
                else:
                    # Single band with colormap
                    data = src.read(1)
                    import matplotlib.cm as cm
                    data_min, data_max = np.nanmin(data), np.nanmax(data)
                    if data_max > data_min:
                        data_norm = (data - data_min) / (data_max - data_min)
                    else:
                        data_norm = np.zeros_like(data)
                    cmap_func = cm.get_cmap(colormap)
                    img_array = cmap_func(data_norm)
                    img_array = (img_array[:, :, :3] * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array)
                
                img_buffer = BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}", bounds_latlon
        
        # Reproject classification masks
        if first_class is not None:
            first_class_binary = (first_class > 0).astype(np.uint8)
            before_class_wgs84_path = reproject_to_wgs84(first_class_binary, f"before_class_{first_month}")
        
        if last_class is not None:
            last_class_binary = (last_class > 0).astype(np.uint8)
            after_class_wgs84_path = reproject_to_wgs84(last_class_binary, f"after_class_{last_month}")
        
        # Reproject change mask
        if change_mask is not None:
            change_mask_wgs84_path = reproject_to_wgs84(change_mask.astype(np.uint8), "change_mask")
        
        # Generate and reproject RGB images
        first_rgb = generate_rgb_from_sentinel(first_image_path)
        last_rgb = generate_rgb_from_sentinel(last_image_path) if last_image_path else None
        
        if first_rgb is not None:
            before_rgb_wgs84_path = reproject_rgb_to_wgs84(first_rgb, f"before_rgb_{first_month}")
        
        if last_rgb is not None:
            after_rgb_wgs84_path = reproject_rgb_to_wgs84(last_rgb, f"after_rgb_{last_month}")
        
        # Calculate bounds for map initialization (prevents reset on internet loss)
        if target_bounds:
            map_bounds = [[target_bounds.bottom, target_bounds.left], 
                         [target_bounds.top, target_bounds.right]]
            # Calculate appropriate zoom level based on bounds
            lat_diff = target_bounds.top - target_bounds.bottom
            lon_diff = target_bounds.right - target_bounds.left
            max_diff = max(lat_diff, lon_diff)
            # Approximate zoom level calculation
            if max_diff > 1:
                zoom_level = 8
            elif max_diff > 0.5:
                zoom_level = 10
            elif max_diff > 0.1:
                zoom_level = 12
            elif max_diff > 0.05:
                zoom_level = 14
            else:
                zoom_level = 15
        else:
            map_bounds = None
            zoom_level = 15
        
        # Create Folium map with explicit bounds to prevent reset
        m = folium.Map(
            location=center, 
            zoom_start=zoom_level, 
            tiles=None,
            # These options help stabilize the map
            prefer_canvas=True,
            zoom_control=True,
            scrollWheelZoom=True,
            dragging=True
        )
        
        # Add fullscreen control
        plugins.Fullscreen(
            position='topleft',
            title='نمایش تمام‌صفحه',
            title_cancel='خروج از نمایش تمام‌صفحه',
            force_separate_button=True
        ).add_to(m)
        
        # BASE LAYERS - mutually exclusive background layers
        # Google Satellite with year in name
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name=f'تصویر ماهواره‌ای گوگل ({last_year})',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Google Maps
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google Maps',
            name='نقشه خیابان‌های گوگل',
            overlay=False,
            control=True
        ).add_to(m)
        
        # OpenStreetMap
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='نقشه باز شهری',
            overlay=False,
            control=True
        ).add_to(m)
        
        # OVERLAY LAYERS - stack on top of base layers
        # Sentinel RGB layers
        if before_rgb_wgs84_path:
            try:
                img_data, bounds = raster_to_folium_overlay(before_rgb_wgs84_path, is_rgb=True)
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=bounds,
                    opacity=0.8,
                    name=f"تصویر رنگی ابتدای دوره ({first_month})",
                    overlay=True,
                    control=True,
                    show=False
                ).add_to(m)
            except Exception as e:
                st.warning("لایه تصویر رنگی ابتدای دوره نمایش داده نشد.")
        
        if after_rgb_wgs84_path:
            try:
                img_data, bounds = raster_to_folium_overlay(after_rgb_wgs84_path, is_rgb=True)
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=bounds,
                    opacity=0.8,
                    name=f"تصویر رنگی پایان دوره ({last_month})",
                    overlay=True,
                    control=True,
                    show=False
                ).add_to(m)
            except Exception as e:
                st.warning("لایه تصویر رنگی پایان دوره نمایش داده نشد.")
        
        # Classification layers
        if before_class_wgs84_path:
            try:
                img_data, bounds = raster_to_folium_overlay(
                    before_class_wgs84_path, colormap='Greens', is_binary=True
                )
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=bounds,
                    opacity=0.7,
                    name=f"ساختمان‌های ابتدای دوره ({first_month}) ـ سبز",
                    overlay=True,
                    control=True,
                    show=True
                ).add_to(m)
            except Exception as e:
                st.warning("لایه ساختمان‌های ابتدای دوره نمایش داده نشد.")
        
        if after_class_wgs84_path:
            try:
                img_data, bounds = raster_to_folium_overlay(
                    after_class_wgs84_path, colormap='Reds', is_binary=True
                )
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=bounds,
                    opacity=0.7,
                    name=f"ساختمان‌های پایان دوره ({last_month}) ـ قرمز",
                    overlay=True,
                    control=True,
                    show=True
                ).add_to(m)
            except Exception as e:
                st.warning("لایه ساختمان‌های پایان دوره نمایش داده نشد.")
        
        # Change detection mask (top overlay) - DARK BLUE
        if change_mask_wgs84_path:
            try:
                img_data, bounds = raster_to_folium_overlay(
                    change_mask_wgs84_path, is_change_mask=True
                )
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=bounds,
                    opacity=0.8,
                    name=f"تغییرات شناسایی‌شده ({first_month} تا {last_month}) ـ آبی",
                    overlay=True,
                    control=True,
                    show=True
                ).add_to(m)
            except Exception as e:
                st.warning("لایه تغییرات شناسایی‌شده نمایش داده نشد.")
        
        # Fit map to bounds and set max bounds to prevent excessive panning
        if target_bounds:
            m.fit_bounds([[target_bounds.bottom, target_bounds.left], 
                         [target_bounds.top, target_bounds.right]])
            # Add a slight padding to max bounds
            padding = 0.01  # degrees
            m.options['maxBounds'] = [
                [target_bounds.bottom - padding, target_bounds.left - padding],
                [target_bounds.top + padding, target_bounds.right + padding]
            ]
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add custom JavaScript to prevent map reset on tile load errors
        custom_js = f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Store initial view
            var initialCenter = [{center[0]}, {center[1]}];
            var initialZoom = {zoom_level};
            
            // Find the map object
            setTimeout(function() {{
                var maps = document.querySelectorAll('.folium-map');
                maps.forEach(function(mapEl) {{
                    if (mapEl._leaflet_map) {{
                        var map = mapEl._leaflet_map;
                        
                        // Prevent tile errors from affecting map state
                        map.on('tileerror', function(e) {{
                            console.log('Tile error (ignored):', e);
                        }});
                        
                        // Store current view on every move
                        var lastCenter = map.getCenter();
                        var lastZoom = map.getZoom();
                        
                        map.on('moveend', function() {{
                            lastCenter = map.getCenter();
                            lastZoom = map.getZoom();
                        }});
                    }}
                }});
            }}, 1000);
        }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(custom_js))
        
        # Presentation only: Persian controls, keeping the map's geographic direction LTR.
        m.get_root().header.add_child(folium.Element("<style>" + _fa_font_css() + """
        html, body, .folium-map { direction: ltr; }
        .leaflet-control-layers, .leaflet-control-attribution {
            font-family: var(--bcd-font);
        }
        .leaflet-control-layers { direction: rtl; text-align: right; line-height: 1.9; font-size: 17px; }
        .leaflet-control-layers label { text-align: right; }
        </style>"""))
        m.get_root().html.add_child(folium.Element("""
        <script>
        document.addEventListener('DOMContentLoaded', function () {
            document.querySelectorAll('.leaflet-control-zoom-in').forEach(function (el) {
                el.title = 'بزرگ‌نمایی'; el.setAttribute('aria-label', 'بزرگ‌نمایی');
            });
            document.querySelectorAll('.leaflet-control-zoom-out').forEach(function (el) {
                el.title = 'کوچک‌نمایی'; el.setAttribute('aria-label', 'کوچک‌نمایی');
            });
        });
        </script>
        """))
        # Display the map using components.html
        map_html = m.get_root().render()
        components.html(map_html, height=700)
        
        st.caption(f"""
        راهنمای لایه‌ها: سبز، ساختمان‌های ابتدای دوره ({first_month})؛
        قرمز، ساختمان‌های پایان دوره ({last_month})؛ آبی، ساختمان‌های جدید شناسایی‌شده.
        تصاویر نقشه پایه ممکن است هم‌زمان با دوره تحلیل نباشند.
        برای نمایش نقشه پایه به اینترنت نیاز است؛ لایه‌های نتایج به‌صورت تعبیه‌شده نمایش داده می‌شوند.
        """)
        
    except Exception as e:
        st.error("نمایش نقشه نتایج با خطا روبه‌رو شد؛ تصاویر و دریافت فایل‌های خروجی را بررسی کنید.")

def change_detection_tab():
    """Change Detection Analysis Tab"""
    st.title("🔍 شناسایی ساخت‌وسازهای جدید")
    
    # Check if we have probability maps
    if not st.session_state.probability_maps:
        st.warning("هنوز تصویری برای تحلیل آماده نیست. ابتدا محدوده و بازهٔ زمانی را در زبانهٔ «انتخاب محدوده و پردازش» انتخاب و پردازش کنید.")
        st.info("پس از پایان پردازش، تنظیمات تحلیل و نتایج تغییرات در این بخش در دسترس خواهند بود.")
        return
    
    n_months = len(st.session_state.probability_maps)
    sorted_months = sorted(st.session_state.probability_maps.keys())
    
    st.success(f"✅ **{n_months} ماه** برای تحلیل آماده است؛ از **{sorted_months[0]}** تا **{sorted_months[-1]}**.")
    
    # =================================================================
    # PARAMETERS SECTION
    # =================================================================
    st.header("⚙️ تنظیمات شناسایی تغییرات")
    
    # Algorithm explanation
    with st.expander("ℹ️ راهنمای معیارهای شناسایی تغییرات", expanded=False):
        st.markdown("""
        ساخت‌وساز جدید زمانی ثبت می‌شود که یک پیکسل:

        - در آغاز دوره غیرساختمان و در پایان دوره ساختمان باشد.
        - تنها یک بار از غیرساختمان به ساختمان تغییر کند و بازگشت وضعیت نداشته باشد.
        - پیش و پس از تغییر، حداقل تداوم تعیین‌شده را رعایت کند.

        مدت تداوم بر اساس تصاویر ماهانهٔ موجود محاسبه می‌شود. آستانه‌ها و مدت‌ها را متناسب با تحلیل خود تعیین کنید.
        """)
    
    # Parameters in two columns with text inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 آستانه‌های احتمال")
        non_building_thr = st.text_input(
            "آستانهٔ غیرساختمان",
            value="0.1",
            help="حد احتمال غیرساختمان، از صفر تا ۰٫۵. در محاسبه، مقدار کمتر یا مساوی این حد برای آغاز دوره پذیرفته می‌شود.",
            key="cd_non_building_thr"
        )
        building_thr = st.text_input(
            "آستانهٔ ساختمان",
            value="0.9",
            help="حد احتمال ساختمان، از ۰٫۵ تا یک. در محاسبه، مقدار بیشتر یا مساوی این حد برای پایان دوره پذیرفته می‌شود.",
            key="cd_building_thr"
        )
    
    with col2:
        st.subheader("⏱️ حداقل تداوم وضعیت")
        min_non_building = st.text_input(
            "حداقل تداوم غیرساختمان (ماه)",
            value="2",
            help="حداقل تعداد تصاویر ماهانهٔ پیش از تغییر که باید وضعیت غیرساختمان داشته باشند.",
            key="cd_min_non_building"
        )
        min_building = st.text_input(
            "حداقل تداوم ساختمان (ماه)",
            value="2",
            help="حداقل تعداد تصاویر ماهانهٔ پس از تغییر که باید وضعیت ساختمان داشته باشند.",
            key="cd_min_building"
        )
    
    # Validate inputs
    try:
        non_building_thr_val = float(non_building_thr)
        building_thr_val = float(building_thr)
        min_non_building_val = int(min_non_building)
        min_building_val = int(min_building)
        
        # Validation checks
        errors = []
        if not (0.0 <= non_building_thr_val <= 0.5):
            errors.append("آستانهٔ غیرساختمان باید بین صفر و ۰٫۵ باشد.")
        if not (0.5 <= building_thr_val <= 1.0):
            errors.append("آستانهٔ ساختمان باید بین ۰٫۵ و یک باشد.")
        if min_non_building_val < 1:
            errors.append("حداقل تداوم غیرساختمان باید یک ماه یا بیشتر باشد.")
        if min_building_val < 1:
            errors.append("حداقل تداوم ساختمان باید یک ماه یا بیشتر باشد.")
        if min_non_building_val + min_building_val > n_months:
            errors.append(f"مجموع مدت تداوم ({min_non_building_val + min_building_val} ماه) از تعداد ماه‌های موجود ({n_months}) بیشتر است.")
        
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            params_valid = False
        else:
            params_valid = True
            st.info(f"تنظیمات تحلیل: آستانهٔ غیرساختمان {non_building_thr_val}، آستانهٔ ساختمان {building_thr_val}؛ حداقل تداوم پیش از تغییر {min_non_building_val} و پس از تغییر {min_building_val} ماه.")
    
    except ValueError as e:
        st.error(f"مقدار پارامتر معتبر نیست. آستانه‌ها را به‌صورت عدد اعشاری و مدت‌ها را به‌صورت عدد صحیح وارد کنید.")
        params_valid = False
    
    # =================================================================
    # RUN CHANGE DETECTION
    # =================================================================
    st.divider()
    
    if params_valid:
        if st.button("🔍 اجرای تحلیل تغییرات", type="primary", use_container_width=True):
            with st.spinner("در حال شناسایی تغییرات ساختمان‌ها در طول زمان…"):
                # Run change detection
                change_mask, timing_map, stats = analyze_building_transition(
                    st.session_state.probability_maps,
                    non_building_thr=non_building_thr_val,
                    building_thr=building_thr_val,
                    non_building_duration=min_non_building_val,
                    building_duration=min_building_val
                )
                
                if change_mask is not None:
                    st.session_state.change_detection_result = {
                        'mask': change_mask,
                        'timing_map': timing_map,
                        'stats': stats,
                        'params': {
                            'non_building_thr': non_building_thr_val,
                            'building_thr': building_thr_val,
                            'min_non_building': min_non_building_val,
                            'min_building': min_building_val
                        }
                    }
                    st.success("شناسایی تغییرات با موفقیت انجام شد.")
                    st.rerun()
                else:
                    st.error("تحلیل تغییرات انجام نشد. تعداد ماه‌های موجود و تنظیمات تداوم را بررسی کنید.")
    
    # =================================================================
    # DISPLAY RESULTS
    # =================================================================
    if st.session_state.change_detection_result:
        result = st.session_state.change_detection_result
        change_mask = result['mask']
        stats = result['stats']
        params = result['params']
        
        st.divider()
        st.header("📈 نتایج شناسایی تغییرات")
        
        # Statistics
        with st.container(key="ui_change_result_metrics"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("پیکسل‌های تغییرکرده", _roi_number(stats['change_pixels'], 0))
            col2.metric("کل پیکسل‌های معتبر", _roi_number(stats['total_pixels'], 0))
            col3.metric("درصد تغییرات", _roi_number(stats['change_percentage'], 2) + '٪')
            col4.metric("ماه‌های تحلیل‌شده", _roi_number(stats['n_months'], 0))
        
        st.info(f"📅 دورهٔ نتایج: **{stats['first_month']}** تا **{stats['last_month']}**")
        
        # Display transition statistics by month
        transition_by_month = stats.get('transition_by_month', {})
        if transition_by_month:
            with st.expander("📊 تعداد پیکسل‌های تغییرکرده در هر ماه", expanded=False):
                for month, count in transition_by_month.items():
                    if count > 0:
                        st.write(f"  • **{month}**: {count:,} پیکسل تغییرکرده")
        
        # =================================================================
        # VISUALIZATION
        # =================================================================
        st.subheader("🖼️ مقایسهٔ تصاویر و نقشهٔ تغییرات")
        
        # Get first and last month images
        first_month = stats['first_month']
        last_month = stats['last_month']
        
        first_image_path = st.session_state.valid_months.get(first_month)
        last_image_path = st.session_state.valid_months.get(last_month)
        
        # Get valid patch bounds for cropping
        valid_mask = st.session_state.valid_patches_mask
        original_size = change_mask.shape
        crop_bounds = get_valid_patch_bounds(valid_mask, PATCH_SIZE, original_size)
        
        # Generate RGB images
        first_rgb = generate_rgb_from_sentinel(first_image_path) if first_image_path else None
        last_rgb = generate_rgb_from_sentinel(last_image_path) if last_image_path else None
        
        # Get first and last classification masks
        first_prob = st.session_state.probability_maps.get(first_month)
        last_prob = st.session_state.probability_maps.get(last_month)
        first_class = (first_prob > 0.5).astype(np.uint8) * 255 if first_prob is not None else None
        last_class = (last_prob > 0.5).astype(np.uint8) * 255 if last_prob is not None else None
        
        if first_rgb is not None and last_rgb is not None and crop_bounds is not None:
            row_start, row_end, col_start, col_end = crop_bounds
            
            # Crop images to valid area
            first_rgb_cropped = first_rgb[row_start:row_end, col_start:col_end, :]
            last_rgb_cropped = last_rgb[row_start:row_end, col_start:col_end, :]
            change_mask_cropped = change_mask[row_start:row_end, col_start:col_end]
            first_class_cropped = first_class[row_start:row_end, col_start:col_end] if first_class is not None else None
            last_class_cropped = last_class[row_start:row_end, col_start:col_end] if last_class is not None else None
            
            _ui_comparison_panel(
                first_rgb_cropped, last_rgb_cropped, change_mask_cropped,
                first_class_cropped, last_class_cropped, first_month, last_month, stats
            )

            # =================================================================
            # DOWNLOAD SECTION
            # =================================================================
            st.subheader("⬇️ دریافت خروجی‌ها")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Save change mask as GeoTIFF with coordinates
                if first_image_path and os.path.exists(first_image_path):
                    try:
                        with rasterio.open(first_image_path) as src:
                            # Get the transform and CRS from the original image
                            out_meta = src.meta.copy()
                            out_meta.update({
                                'count': 1,
                                'dtype': 'uint8',
                                'height': change_mask.shape[0],
                                'width': change_mask.shape[1]
                            })
                            
                            # Create GeoTIFF in memory
                            geotiff_buffer = BytesIO()
                            with rasterio.open(geotiff_buffer, 'w', **out_meta) as dst:
                                dst.write(change_mask.astype(np.uint8), 1)
                            
                            geotiff_buffer.seek(0)
                            st.download_button(
                                label="📥 دریافت نقشهٔ تغییرات (GeoTIFF)",
                                data=geotiff_buffer.getvalue(),
                                file_name=f"change_detection_{first_month}_to_{last_month}.tif",
                                mime="image/tiff"
                            )
                    except Exception as e:
                        st.error("آماده‌سازی فایل مکانی انجام نشد؛ نسخهٔ تصویری برای دریافت ارائه می‌شود.")
                        # Fallback to PNG
                        change_img = Image.fromarray((change_mask_cropped * 255).astype(np.uint8))
                        buf = BytesIO()
                        change_img.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="📥 دریافت نقشهٔ تغییرات (PNG)",
                            data=buf.getvalue(),
                            file_name=f"change_detection_{first_month}_to_{last_month}.png",
                            mime="image/png"
                        )
            
            with col2:
                # Download cropped change mask as PNG for quick viewing
                change_img = Image.fromarray((change_mask_cropped * 255).astype(np.uint8))
                buf = BytesIO()
                change_img.save(buf, format='PNG')
                buf.seek(0)
                st.download_button(
                    label="📥 دریافت پیش‌نمایش برش‌خورده (PNG)",
                    data=buf.getvalue(),
                    file_name=f"change_preview_{first_month}_to_{last_month}.png",
                    mime="image/png"
                )
            
            with col3:
                # Save parameters and stats as text
                report_text = f"""Change Detection Report
================================
Time Range: {first_month} to {last_month}
Months Analyzed: {stats['n_months']}

Parameters:
- Non-building threshold: {params['non_building_thr']}
- Building threshold: {params['building_thr']}
- Min non-building duration: {params['min_non_building']} months
- Min building duration: {params['min_building']} months

Results:
- Change pixels: {stats['change_pixels']:,}
- Total valid pixels: {stats['total_pixels']:,}
- Change rate: {stats['change_percentage']:.4f}%
"""
                st.download_button(
                    label="📄 دریافت گزارش (TXT)",
                    data=report_text,
                    file_name=f"change_detection_report_{first_month}_to_{last_month}.txt",
                    mime="text/plain"
                )
            
            # =================================================================
            # INTERACTIVE MAP
            # =================================================================
            st.divider()
            st.subheader("🗺️ نقشهٔ تعاملی تغییرات")
            
            display_interactive_map(
                first_image_path, last_image_path,
                first_class, last_class, change_mask,
                first_month, last_month, crop_bounds, original_size
            )
        
        else:
            st.error("نمایش نتایج ممکن نیست؛ تصویر رنگ طبیعی یا محدودهٔ معتبر تصویر در دسترس نیست.")


if __name__ == "__main__":
    main()
