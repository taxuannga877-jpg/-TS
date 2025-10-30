import os
import argparse
import numpy as np
from rmsd import calculate_rmsd, kabsch_rmsd, get_coordinates_xyz

def calculate_rmsd_with_alignment(ts_path, ts_pred_path):
    try:
        # Note: Use the second return value as coordinates!
        _, P = get_coordinates_xyz(ts_pred_path)  # Coordinates of predicted structure
        _, Q = get_coordinates_xyz(ts_path)       # Coordinates of reference structure

        # Data validation
        if not isinstance(P, np.ndarray) or not isinstance(Q, np.ndarray):
            raise ValueError("Coordinate data is not a valid numpy array")

        if P.size == 0 or Q.size == 0:
            raise ValueError("Coordinate array is empty")

        if P.shape[1] != 3 or Q.shape[1] != 3:
            raise ValueError(f"Invalid coordinate array shape: {P.shape} vs {Q.shape}, expected Nx3")

        if P.shape[0] != Q.shape[0]:
            raise ValueError(f"Atom count mismatch: {P.shape[0]} vs {Q.shape[0]}")

        # Ensure data types are consistent
        P = P.astype(np.float64)
        Q = Q.astype(np.float64)

        # Calculate RMSD
        rmsd_value = kabsch_rmsd(P, Q)
        return rmsd_value, True  # Return success flag
    except Exception as e:
        print(f"Error calculating RMSD [{ts_pred_path} vs {ts_path}]: {e}")
        return float('inf'), False  # Return infinity as error RMSD value

def calculate_rmsd_score(rmsd_value):
    """根据RMSD值计算分数（最高40分）"""
    if rmsd_value < 0.2:
        return 40
    elif rmsd_value > 0.5:
        return 0
    else:
        # 线性计算0.2到0.5之间的分数
        return 40 - ((rmsd_value - 0.2) / 0.3) * 40

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD error and scores between TS.xyz and TS_pred.xyz using rmsd package')
    parser.add_argument('--ts_dir', default='GT', help='Path to directory containing TS.xyz files')
    parser.add_argument('--ts_pred_dir',default='baseline', help='Path to directory containing TS_pred.xyz files')
    parser.add_argument('--prefix', default='rxn', help='Folder prefix for matching reaction pairs')
    parser.add_argument('--output', default='rmsd_results.csv', help='Output CSV file name for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='RMSD threshold (Å) for accuracy statistics')
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.isdir(args.ts_dir):
        print(f"Error: Directory {args.ts_dir} does not exist")
        return
    if not os.path.isdir(args.ts_pred_dir):
        print(f"Error: Directory {args.ts_pred_dir} does not exist")
        return

    # Get all prefix-matched folders
    ts_folders = [f for f in os.listdir(args.ts_dir)
                 if os.path.isdir(os.path.join(args.ts_dir, f)) and f.startswith(args.prefix)]
    ts_pred_folders = [f for f in os.listdir(args.ts_pred_dir)
                      if os.path.isdir(os.path.join(args.ts_pred_dir, f)) and f.startswith(args.prefix)]

    if not ts_folders or not ts_pred_folders:
        print(f"No folders found with prefix {args.prefix}")
        return

    # Create folder name to path mappings
    ts_folder_map = {f: os.path.join(args.ts_dir, f) for f in ts_folders}
    ts_pred_folder_map = {f: os.path.join(args.ts_pred_dir, f) for f in ts_pred_folders}

    # Find common folder names
    common_folders = set(ts_folder_map.keys()) & set(ts_pred_folder_map.keys())

    if not common_folders:
        print("No matching folder pairs found in both directories")
        return

    print(f"Found {len(common_folders)} matching folder pairs")
    results = []
    total_pairs = len(common_folders)

    # Iterate over each matching folder pair to calculate RMSD
    for folder in sorted(common_folders):
        ts_path = os.path.join(ts_folder_map[folder], 'TS.xyz')
        ts_pred_path = os.path.join(ts_pred_folder_map[folder], 'TS_pred.xyz')

        # Check if files exist
        file_missing = False
        if not os.path.exists(ts_path):
            print(f"Warning: TS.xyz file missing in {ts_folder_map[folder]}")
            file_missing = True
        if not os.path.exists(ts_pred_path):
            print(f"Warning: TS_pred.xyz file missing in {ts_pred_folder_map[folder]}")
            file_missing = True

        if file_missing:
            results.append((folder, float('inf'), False))
            continue

        # Calculate RMSD (with alignment)
        rmsd_value, success = calculate_rmsd_with_alignment(ts_path, ts_pred_path)
        results.append((folder, rmsd_value, success))

        status = "Success" if success and rmsd_value <=0.5 else "Failure"
        print(f"{folder}: RMSD = {rmsd_value:.4f} Å ({status})")

    # Save results to CSV file (including failed records)
    with open(args.output, 'w') as f:
        f.write("Folder,RMSD,Status\n")
        for folder, rmsd_val, success in results:
            status = "Success" if success and rmsd_val <=0.5 else "Failure"
            f.write(f"{folder},{rmsd_val:.6f},{status}\n")
    print(f"Results saved to {args.output}")

    # Statistics module
    rmsd_values = np.array([r[1] for r in results])
    success_count = np.sum([r[2] for r in results])
    accurate_count = np.sum(rmsd_values < args.threshold)
    accurate_ratio = accurate_count / total_pairs * 100

    # Calculate average error for successful and accurate predictions (RMSD < threshold)
    valid_rmsd = [r[1] for r in results if r[2] and r[1] < args.threshold]
    accurate_avg = np.mean(valid_rmsd) if valid_rmsd else 0

    # 使用符合条件的RMSD平均值来计算分数
    if valid_rmsd:
        rmsd_score = calculate_rmsd_score(accurate_avg)
    else:
        rmsd_score = 0

    # 计算成功率分数（使用RMSD < threshold的比例）
    success_rate = accurate_count / total_pairs
    success_score = success_rate * 30

    # 计算总分
    total_score = rmsd_score + success_score

    print("\n===== Statistics =====")
    print(f"Total reaction pairs: {total_pairs}")
    print(f"Successful RMSD calculations: {success_count} ({success_count/total_pairs*100:.2f}%)")
    print(f"Reactions with RMSD < {args.threshold} Å: {accurate_count} ({accurate_ratio:.2f}%)")
    if valid_rmsd:
        print(f"Average error for successful predictions with RMSD < {args.threshold} Å: {accurate_avg:.4f} Å")
    
    print("\n===== Scores =====")
    print(f"Selected RMSD: {accurate_avg:.4f} Å (Only successful predictions with RMSD < {args.threshold} Å)")
    print(f"RMSD Score: {rmsd_score:.2f}/40")
    print(f"Success Rate Score: {success_score:.2f}/30 (Success Rate: {success_rate*100:.2f}%)")
    print(f"Total Score: {total_score:.2f}/70")

if __name__ == "__main__":
    main()