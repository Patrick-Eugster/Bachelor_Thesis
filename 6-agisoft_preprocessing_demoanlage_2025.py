import datetime
import logging
from pathlib import Path
from time import time
from typing import Dict, List

import Metashape
import pandas as pd

MARKERS_PER_FIELD = 6 # Number of markers per field, used to determine if all markers are present

# Set up logging
log_folder = Path('logs')
log_folder.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_folder / f'{Path(__file__).stem}-{timestamp}.log'
logging.basicConfig(filename=log_file, level=logging.INFO)


def write_markers_coordinates_per_plot(marker_paths: Dict[str, Path], df: pd.DataFrame, columns: List[str], overwrite=False) -> None:
    """Write marker coordinates to text files for each plot.
    
    Args:
        marker_paths: Dictionary mapping plot IDs to output file paths
        df: DataFrame containing marker coordinates
        columns: List of column names for coordinates (e.g., ['Easting', 'Northing', 'Elevation'])
        overwrite: Whether to overwrite existing files
    """
    # Get coordinates and write to files
    for field_id, path in marker_paths.items():
        # Skip if file exists and overwrite is False
        if path.exists() and not overwrite:
            print(f'File {path} already exists. Skipping...')
            continue
        
        # Get coordinates for this plot
        plot_data = df[df['field_id'] == field_id]

        if plot_data.empty:
            print(f'WARNING: No GPS data found for field {field_id}, skipping...')
            continue
        
        marker_positions = plot_data[columns].values

        if len(marker_positions) != MARKERS_PER_FIELD:
            print(f'WARNING: Found GPS data for only {len(marker_positions)} markers in field {field_id}, but there should be {MARKERS_PER_FIELD}')

        with open(path, 'w') as f:
            # Write header with coordinate system info
            f.write('# CoordinateSystem: PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",DATUM["CH1903+",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[674.374,15.056,405.346,0,0,0,0],AUTHORITY["EPSG","6150"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9102"]],AUTHORITY["EPSG","4150"]],PROJECTION["Oblique_Mercator",AUTHORITY["EPSG","9815"]],PARAMETER["latitude_of_center",46.95240555555561],PARAMETER["longitude_of_center",7.439583333333329],PARAMETER["azimuth",90],PARAMETER["recitified_grid_angle",90],PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],PARAMETER["false_northing",1200000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","2056"]]\n')
            f.write("#Label\tX/Easting\tY/Northing\tZ/Altitude\n")
            
            # Write markers coordinates
            for _ , marker_id, x, y, z in marker_positions:
                f.write(f"target {marker_id},{x},{y},{z}\n")

            print(f'Wrote {len(marker_positions)} marker coordinates for field {field_id} to {path}')


def process_markers(markers_path: Path, all_fields: List[str]) -> Dict[str, Path]:
    """Process marker coordinates from a CSV file and write them to text files for each field."""
    if not markers_path.exists():
        print(f"Marker file {markers_path} is missing, skipping marker processing.")
        return

    markers_paths = {field: markers_path.parent / f'{field}_coordinates.txt' for field in sorted(all_fields)}
    if not all([path.exists() for path in markers_paths.values()]):
        print(f"Loading marker coordinates from {markers_path}")
        columns = ['field_marker_id', 'Easting', 'Northing', 'Elevation']
        markers_df = pd.read_csv(markers_path, header=None, names=columns, usecols=[0, 1, 2, 3])

        # Check if values in this column match the expected format (e.g., "A1", "B2", "C3")
        if markers_df['field_marker_id'].str.match(r'^[A-Z]\d+$').any():
            # Extract field letter and marker number
            markers_df['field_letter'] = markers_df['field_marker_id'].str[0]
            markers_df['marker_id'] = markers_df['field_marker_id'].str[1:]
            markers_df['field_id'] = 'field_' + markers_df['field_letter']
            markers_df.drop(columns=['field_marker_id', 'field_letter'], inplace=True)
        else:
            print("Warning: Could not find field-marker identifier in CSV file")
            print("CSV columns:", markers_df.columns.tolist())
            print("First few rows:", markers_df.head())
            return markers_paths
    
        # Write marker coordinates
        write_markers_coordinates_per_plot(
            markers_paths,
            markers_df,
            columns=['field_id', 'marker_id'] + columns[1:],
            overwrite=False
        )
    return markers_paths

def run_agisoft_SfM_pipeline(images_folder_path: Path, save_path: Path, markers_path: Path) -> None:
    
    # 1. Initialize Agisoft Metashape
    doc = Metashape.Document() # Metashape.app.document

    # 2. Add photos
    chunk = doc.addChunk()
    images = [str(image_path) for image_path in images_folder_path.glob('*.jpg')]
    chunk.addPhotos(images)

    # 3. Detect markers
    chunk.detectMarkers(tolerance=100)

    # 4 Convert Cameras to Swiss coordinate system (see https://www.agisoft.com/forum/index.php?topic=9250.0)
    crs = Metashape.CoordinateSystem('PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",DATUM["CH1903+",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[674.374,15.056,405.346,0,0,0,0],AUTHORITY["EPSG","6150"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9102"]],AUTHORITY["EPSG","4150"]],PROJECTION["Oblique_Mercator",AUTHORITY["EPSG","9815"]],PARAMETER["latitude_of_center",46.95240555555561],PARAMETER["longitude_of_center",7.439583333333329],PARAMETER["azimuth",90],PARAMETER["recitified_grid_angle",90],PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],PARAMETER["false_northing",1200000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","2056"]]')
    for camera in chunk.cameras:
        if camera.reference.location:
            camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location, chunk.crs, crs)
    chunk.crs = crs
    chunk.updateTransform()

    # 5. Import markers coordinates
    chunk.importReference(str(markers_path), format=Metashape.ReferenceFormatCSV, delimiter=',', columns='nxyz', crs=crs)
    
    # 6. Align images
    chunk.matchPhotos(downscale=0, reference_preselection_mode=Metashape.ReferencePreselectionSequential) # 0 - Highest; 1 - High
    chunk.alignCameras()

    # 7. Export: save project and export colmap cameras
    save_path.mkdir(exist_ok=True, parents=True)
    doc.save(str(save_path / f"{save_path.name}.psx"))
    
    colmap_path = save_path / 'colmap' / 'colmap_info.txt'
    colmap_path.parent.mkdir(parents=True, exist_ok=True)
    chunk.exportCameras(str(colmap_path), format=Metashape.CamerasFormatColmap, convert_to_pinhole=True, save_markers=True)

def process_data(images_base_path: Path, processed_base_path: Path, markers_csv_path: Path = None):
    """Process all FIP data.
    
    Args:
        raw_base_path: Path to the raw data directory
        processed_base_path: Path to the processed data directory
        markers_csv_path: Path to the CSV file containing marker coordinates
    """
    total_start_time = time()

    processed_base_path.mkdir(parents=True, exist_ok=True)
    all_fields: List[str] = sorted([field.name for field in (images_base_path).iterdir() if field.is_dir()])
    logging.info(f"Found {len(all_fields)} unique fields: {all_fields}")

    # Process markers
    markers_path = process_markers(markers_csv_path, all_fields)

    # Process images for each date
    for field in all_fields:
        field_start_time = time()
        logging.info(f"Processing field: {field}")
        images_field_path = images_base_path / field
        processed_field_dir = processed_base_path / field
        processed_field_dir.mkdir(exist_ok=True)

        dates = sorted([date.name for date in images_field_path.iterdir() if date.is_dir()])
        processed_dates = []
        for date in sorted(dates):
            date_start_time = time()
            logging.info(f"Processing date: {date}:")
            images_date_path = images_field_path / date / "OpenCamera" / "images"
            date = date.split('_')[0] # discard extra information from folder name
            processed_date_dir = processed_field_dir / date
            save_path = processed_field_dir / date
            print(images_date_path)
            if not images_date_path.exists():
                logging.warning(f"No images found for date {date} in field {field}, skipping...")
                continue

            # Agisoft processing
            agisoft_folder = processed_date_dir / "agisoft"
            if not agisoft_folder.exists() or not (agisoft_folder / f"{agisoft_folder.name}.psx" ).exists():
                agisoft_folder.mkdir(parents=True, exist_ok=True)
                run_agisoft_SfM_pipeline(images_date_path, agisoft_folder, markers_path.get(field, None))
                logging.info(f'Processing finished in {time() - date_start_time} seconds. Saved camera poses, point cloud and mesh in {save_path}')
                processed_dates.append(date)
            else: 
                logging.info(f"Field {field} already processed, skipping Agisoft processing")

        logging.info(f'Finished processing {field}. Processed {len(processed_dates)} dates ({processed_dates}) in {time() - field_start_time} seconds')

    logging.info(f'Processed all fields in {time() - total_start_time} seconds')           

def main():
    # Define paths
    VERSION = 0
    base_path = Path("/home/jgajardo/public-mnt/Evaluation/Projects/KP0034_jgajardo/data/wheat_smartphone_images/strickhof_fields/2025/demoanlage")
    images_path = base_path / "images" 
    processed_path = base_path / "processed" / f"v{VERSION}"
    markers_csv_path = base_path / "metadata" / "markers" / "joaquin-20250325.csv"
    
    # Process Demoanlage 2025 data
    process_data(images_path, processed_path, markers_csv_path)

    logging.info("All Demoanlage 2025 images have been processed!")

if __name__ == "__main__":
    main()