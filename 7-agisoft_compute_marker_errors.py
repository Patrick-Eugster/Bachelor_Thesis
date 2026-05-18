import argparse
import math
from pathlib import Path

import Metashape
import pandas as pd


def load_reference_distances(excel_path: Path) -> dict:
    """Parse upper-triangular distance matrix sheets (cm) into a nested dict.

    Returns: {field_name: {frozenset({label1, label2}): distance_m}}
    Sheet names like 'plot A' map to field names like 'field_A'.
    Marker index N (1-based) maps to label 'target N'.
    """
    xl = pd.ExcelFile(excel_path)
    ref = {}
    for sheet in xl.sheet_names:
        field = "field_" + sheet.split()[-1]
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        col_indices = df.iloc[0, 1:].tolist()
        row_indices = df.iloc[1:, 0].tolist()
        pairs = {}
        for r_idx, row_num in enumerate(row_indices):
            for c_idx, col_num in enumerate(col_indices):
                val = df.iloc[r_idx + 1, c_idx + 1]
                if pd.notna(val) and val > 0:
                    label1 = f"target {int(row_num)}"
                    label2 = f"target {int(col_num)}"
                    pairs[frozenset({label1, label2})] = val / 100.0  # cm -> m
        ref[field] = pairs
    return ref


def compute_project_statistics(project_path: Path, verbose: bool = False, ref_distances: dict = None) -> tuple:
    doc = Metashape.Document()
    doc.open(str(project_path))
    chunk = doc.chunk
    date = project_path.parent.parent.name
    field = project_path.parent.parent.parent.name

    project_data = []
    marker_data = []

    for chunk in doc.chunks:
        if not chunk.transform or not chunk.markers:
            continue

        transform = chunk.transform
        crs = chunk.crs
        if chunk.marker_crs:
            crs = chunk.marker_crs
        ecef_crs = crs.geoccs if crs and crs.geoccs else Metashape.CoordinateSystem("LOCAL")

        aligned_cameras = sum(1 for cam in chunk.cameras if cam.transform)
        tie_point_count = len(chunk.tie_points.points) if chunk.tie_points else 0

        if verbose:
            print(f"\nProcessing date {date} in field {field} for chunk {chunk.label}")
            print(f"Aligned Cameras: {aligned_cameras}/{len(chunk.cameras)}")
            print(f"Tie Points: {tie_point_count}")
            print("Marker Errors:")
            print(f"{'Label':<20} {'3D Error (m)':>15} {'Reproj Error (px)':>20}")

        project_data.append({
            "Date": date,
            "Field": field,
            "Chunk": chunk.label,
            "Cameras": len(chunk.cameras),
            "Aligned Cameras": aligned_cameras,
            "Tie Points": tie_point_count
        })

        for marker in chunk.markers:
            if not marker.position:
                continue

            error_3d = None
            if marker.reference.location:
                pos_est = Metashape.CoordinateSystem.transform(transform.matrix.mulp(marker.position), ecef_crs, crs)
                pos_ref = marker.reference.location
                pos_est_ecef = Metashape.CoordinateSystem.transform(pos_est, crs, ecef_crs)
                pos_ref_ecef = Metashape.CoordinateSystem.transform(pos_ref, crs, ecef_crs)
                error_vec = pos_est_ecef - pos_ref_ecef
                error_3d = error_vec.norm()

            # Reprojection error
            error_sq_sum = 0
            count = 0
            for cam, proj in marker.projections.items():
                if not cam.transform:
                    continue
                reproj = cam.project(marker.position)
                if reproj:
                    error_px = (reproj - proj.coord[:2]).norm()
                    error_sq_sum += error_px ** 2
                    count += 1

            rms_error = math.sqrt(error_sq_sum / count) if count > 0 else None

            if verbose:
                error_3d_str = f"{error_3d:.3f} m" if error_3d is not None else "N/A"
                error_px_str = f"{rms_error:.3f} px" if rms_error is not None else "N/A"
                print(f"{marker.label:<20} {error_3d_str:>15} {error_px_str:>20}")
            marker_data.append({
                "Date": date,
                "Field": field,
                "Chunk": chunk.label,
                "Marker": marker.label,
                "3D Error (m)": round(error_3d, 4) if error_3d is not None else None,
                "Reproj Error (px)": round(rms_error, 4) if rms_error is not None else None,
                "Num Projections": count,
            })

        # Pairwise distance errors vs ruler measurements
        if ref_distances:
            field_refs = ref_distances.get(field, {})
            marker_map = {m.label: m for m in chunk.markers if m.position}
            scale = chunk.transform.scale or 1.0
            dist_errors = []
            if verbose and field_refs:
                print(f"\nDistance Errors:")
                print(f"{'Pair':<25} {'True (m)':>10} {'Estimated (m)':>15} {'Error (m)':>12}")
            for pair, true_dist in field_refs.items():
                label1, label2 = sorted(pair)
                m1 = marker_map.get(label1)
                m2 = marker_map.get(label2)
                if m1 is None or m2 is None:
                    continue
                estimated_dist = (m1.position - m2.position).norm() * scale
                dist_error = estimated_dist - true_dist
                dist_errors.append(dist_error)
                if verbose:
                    print(f"{label1+' <-> '+label2:<25} {true_dist:>10.4f} {estimated_dist:>15.4f} {dist_error:>+12.4f}")
            project_data[-1]["Num Distance Pairs"] = len(dist_errors)
            project_data[-1]["Avg Distance Error (m)"] = round(sum(abs(e) for e in dist_errors) / len(dist_errors), 4) if dist_errors else None

    if verbose:
        print(f"{'Average':<20} {sum(m['3D Error (m)'] for m in marker_data if isinstance(m['3D Error (m)'], float)) / len(marker_data):>15.3f} m "
              f"{sum(m['Reproj Error (px)'] for m in marker_data if isinstance(m['Reproj Error (px)'], float)) / len(marker_data):>20.3f} px")

    return project_data, marker_data


def already_processed(summary_file: Path):
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        return set(tuple(row) for row in df[["Date", "Field"]].drop_duplicates().values)
    return set()


def write_results(base_path: Path, new_project_data, new_marker_data):
    full_csv = base_path / "marker_errors_full.csv"
    summary_csv = base_path / "marker_errors_summary.csv"

    # Load previous data
    df_existing = pd.read_csv(full_csv) if full_csv.exists() else pd.DataFrame()
    summary_existing = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()

    df_new = pd.DataFrame(new_marker_data)
    project_df = pd.DataFrame(new_project_data)

    # Group stats
    summary_stats = df_new.groupby(["Date", "Field", "Chunk"]).agg({
        "3D Error (m)": ["mean", "min", "max"],
        "Reproj Error (px)": ["mean", "min", "max"],
        "Num Projections": ["mean", "min", "max"],
    }).reset_index()

    summary_stats.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in summary_stats.columns]
    summary_df = pd.merge(project_df, summary_stats, on=["Date", "Field", "Chunk"], how="left")

    # Append and save
    full_df_out = pd.concat([df_existing, df_new], ignore_index=True)
    summary_out = pd.concat([summary_existing, summary_df], ignore_index=True)

    # Sort by Date and Field before saving
    full_df_out = full_df_out.sort_values(['Date', 'Field', 'Chunk'], ascending=[True, True, True])
    summary_out = summary_out.sort_values(['Date', 'Field', 'Chunk'], ascending=[True, True, True])

    full_df_out.to_csv(full_csv, index=False)
    summary_out.to_csv(summary_csv, index=False)

    print(f"\nSaved:\n- Full: {full_csv}\n- Summary: {summary_csv}")


def process_folder(base_path: Path, verbose=False, ref_distances: dict = None):
    summary_file = base_path / "marker_errors_summary.csv"
    processed = already_processed(summary_file)

    all_markers = []
    all_projects = []

    fields = sorted([f for f in base_path.iterdir() if f.is_dir()])
    for field in fields:
        dates = sorted([d for d in field.iterdir() if d.is_dir()])
        for date in dates:
            fname = date / "agisoft" / "agisoft.psx"
            if not fname.is_file():
                continue
            try:
                id_tuple = (int(date.name), field.name)
                if id_tuple in processed:
                    continue
                project_data, marker_data = compute_project_statistics(fname, verbose=verbose, ref_distances=ref_distances)
                all_projects.extend(project_data)
                all_markers.extend(marker_data)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

    if all_projects and all_markers:
        write_results(base_path, all_projects, all_markers)
    else:
        print("No new data to write.")


def process_single_file(file_path: Path, verbose: bool = True, ref_distances: dict = None) -> None:
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return
    try:
        print(f"Processing file: {file_path}")
        compute_project_statistics(file_path, verbose=verbose, ref_distances=ref_distances)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process Agisoft projects for marker error statistics.")
    parser.add_argument("--path", type=str, help="Path to folder or .psx project file.", default="/home/jgajardo/public-mnt/Evaluation/Projects/KP0034_jgajardo/data/my_data/wheat_smartphone_images/strickhof_fields/2025/demoanlage/processed/v0")
    parser.add_argument("--distances", type=str, help="Path to Excel file with ruler-measured marker distances.", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    ref_distances = load_reference_distances(Path(args.distances)) if args.distances else None

    input_path = Path(args.path)
    if input_path.is_file() and input_path.suffix.lower() == ".psx":
        process_single_file(input_path, verbose=args.verbose, ref_distances=ref_distances)
    elif input_path.is_dir():
        process_folder(input_path, verbose=args.verbose, ref_distances=ref_distances)
    else:
        print("Invalid path. Must be a .psx file or a directory.")


if __name__ == "__main__":

    main()