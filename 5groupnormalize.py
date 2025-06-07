import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Matplotlib setup (no specific font for English, default should work)
plt.rcParams['axes.unicode_minus'] = False

def read_xls_columns_to_arrays(filepath):
    column_a_name = 'Time (s)'
    column_c_name = 'Acceleration y (m/s^2)'
    filename_for_error = os.path.basename(filepath)
    try:
        df = pd.read_excel(filepath, usecols=[column_a_name, column_c_name], engine='xlrd')
        time_array = df[column_a_name].astype(float).tolist()
        acceleration_array = df[column_c_name].astype(float).tolist()
        return time_array, acceleration_array
    except Exception as e:
        print(f"  Error (File: {filename_for_error}): Failed to read or convert data. Check column names and content. Details: {e}")
        return None, None

def find_event_start_time(time_data, accel_data, filename,
                          pre_event_samples_ratio=0.1, min_pre_event_samples=30,
                          threshold_factor=5, min_abs_threshold=0.15):
    if not time_data or not accel_data or len(time_data) < min_pre_event_samples:
        print(f"  Warning ({filename}): Insufficient data to detect event start, using 0.0s.")
        return 0.0, 0 # Return start time and start index
    
    num_samples = len(accel_data)
    num_pre_event_samples = min(max(min_pre_event_samples, int(num_samples * pre_event_samples_ratio)), num_samples - 1)
    if num_pre_event_samples <= 0: return 0.0, 0

    baseline_accel = np.array(accel_data[:num_pre_event_samples])
    baseline_std = np.std(baseline_accel)
    effective_threshold = max(baseline_std * threshold_factor, min_abs_threshold)
    if baseline_std < 1e-6 and min_abs_threshold == 0: effective_threshold = 0.05
    elif effective_threshold < 1e-6: effective_threshold = min_abs_threshold if min_abs_threshold > 0 else 0.05
    
    # print(f"  ({filename}): Baseline std={baseline_std:.4f}, Effective threshold={effective_threshold:.4f}")
    
    event_start_time = time_data[0]
    event_start_index = 0
    for i in range(num_pre_event_samples, num_samples):
        if abs(accel_data[i]) > effective_threshold:
            event_start_time = time_data[i]
            event_start_index = i
            # print(f"  ({filename}): Event detected at time {event_start_time:.4f}s (index {event_start_index})")
            return event_start_time, event_start_index
            
    # print(f"  Warning ({filename}): No significant event detected. Using first data point at time {time_data[0]:.4f}s.")
    return event_start_time, event_start_index

def normalize_acceleration_amplitude(time_data, accel_data, event_start_index, normalization_window_duration=1.0):
    """
    Normalizes the acceleration data based on the max absolute amplitude
    within a window after the event start.
    Returns normalized acceleration_data.
    """
    if not accel_data: return []
    if event_start_index >= len(time_data): return accel_data # Should not happen

    # Define the window to find the max amplitude for normalization
    # Window starts from event_start_index
    window_end_time = time_data[event_start_index] + normalization_window_duration
    
    normalization_window_accel_values = []
    for i in range(event_start_index, len(time_data)):
        if time_data[i] <= window_end_time:
            normalization_window_accel_values.append(accel_data[i])
        else:
            break # Exceeded window duration
            
    if not normalization_window_accel_values: # If window is empty or too short
        # Fallback: use a small number of points after event_start_index
        fallback_points = min(10, len(accel_data) - event_start_index)
        if fallback_points > 0:
            normalization_window_accel_values = accel_data[event_start_index : event_start_index + fallback_points]
        else: # No points after event start, cannot normalize
             print(f"  Warning: Could not find points in normalization window for a dataset. Normalization might be skipped or use full series max.")
             # As a last resort, use max of entire series if window is problematic
             if accel_data:
                max_abs_accel_for_norm = np.max(np.abs(accel_data))
                if max_abs_accel_for_norm > 1e-6: # Avoid division by zero
                    return (np.array(accel_data) / max_abs_accel_for_norm).tolist()
             return accel_data


    if not normalization_window_accel_values: # Still no values
        print(f"  Warning: Normalization window empty, cannot determine normalization factor. Returning original data.")
        return accel_data

    max_abs_accel_for_norm = np.max(np.abs(normalization_window_accel_values))

    if max_abs_accel_for_norm < 1e-6: # Avoid division by zero or near-zero
        # print(f"  Warning: Max abs acceleration in normalization window is near zero ({max_abs_accel_for_norm:.2e}). Skipping normalization for this dataset.")
        return accel_data # Return original if max is too small to normalize meaningfully

    normalized_accel = (np.array(accel_data) / max_abs_accel_for_norm).tolist()
    # print(f"  Normalized with factor: {max_abs_accel_for_norm:.4f}")
    return normalized_accel


def calculate_device_average_normalized_curve(aligned_and_normalized_datasets, common_time_axis_params):
    """
    Calculates the average curve for a single device from its aligned AND NORMALIZED datasets.
    Returns (common_rel_time_axis, mean_normalized_accel) or (None, None).
    """
    if not aligned_and_normalized_datasets: return None, None

    all_interp_norm_accel = []
    min_rel_time_device = float('inf')
    max_rel_time_device = float('-inf')

    for time_adj, _, _, _, _ in aligned_and_normalized_datasets: # time_adj, norm_accel, filename, t_start, event_idx
        if time_adj:
            min_rel_time_device = min(min_rel_time_device, min(time_adj))
            max_rel_time_device = max(max_rel_time_device, max(time_adj))

    if min_rel_time_device == float('inf') or max_rel_time_device == float('-inf') or min_rel_time_device >= max_rel_time_device:
        print(f"  Error: Device data cannot determine a valid common time axis range for averaging.")
        return None, None

    num_points_common_axis = common_time_axis_params.get('num_points', 1000)
    common_rel_time_axis = np.linspace(min_rel_time_device, max_rel_time_device, num_points_common_axis)

    for time_adj, norm_accel, filename, _, _ in aligned_and_normalized_datasets:
        if not time_adj or not norm_accel: continue
        try:
            interp_norm_accel = np.interp(common_rel_time_axis, time_adj, norm_accel)
            all_interp_norm_accel.append(interp_norm_accel)
        except Exception as e:
            print(f"  Error: Interpolation failed for file {filename} during device averaging: {e}")
            continue
            
    if not all_interp_norm_accel: return None, None

    mean_normalized_accel = np.mean(np.array(all_interp_norm_accel), axis=0)
    return common_rel_time_axis, mean_normalized_accel

def plot_all_devices_normalized_comparison(all_device_norm_averages, output_filename, output_directory="OUTPUT_PLOTS"):
    if not all_device_norm_averages:
        print("No device average normalized data to plot.")
        return

    if not os.path.exists(output_directory):
        try: os.makedirs(output_directory)
        except OSError as e: print(f"Error: Cannot create directory '{output_directory}': {e}"); return

    png_filepath = os.path.join(output_directory, output_filename)
    plt.figure(figsize=(14, 8))
    
    global_min_x = float('inf')
    global_max_x = float('-inf')

    for device_id, (time_axis, _) in all_device_norm_averages.items():
        if time_axis is not None and len(time_axis) > 0:
            global_min_x = min(global_min_x, np.min(time_axis))
            global_max_x = max(global_max_x, np.max(time_axis))

    for device_id, (time_axis, mean_norm_accel_curve) in all_device_norm_averages.items():
        if time_axis is not None and mean_norm_accel_curve is not None:
            plt.plot(time_axis, mean_norm_accel_curve, marker='.', linestyle='-', markersize=3, linewidth=1.5, label=f'Device {device_id}')
        else:
            print(f"Warning: Normalized average data missing for device {device_id}, cannot plot.")

    plt.xlabel('Relative Time (s) [Event Start Aligned at t\'=0]')
    plt.ylabel('Normalized Mean Acceleration y (Amplitude)')
    plt.title('Comparison of Normalized Mean Accelerations for All Devices')
    plt.grid(True)
    plt.legend(fontsize='medium')
    
    if global_min_x != float('inf') and global_max_x != float('-inf'):
         plt.xlim(global_min_x, global_max_x)

    # Y-axis for normalized data is typically between -1 and 1, but can be adjusted
    plt.ylim(-1.2, 1.2) # Common for normalized amplitude plots

    try:
        plt.savefig(png_filepath, dpi=200)
        plt.close()
        print(f"Successfully saved all devices normalized comparison chart to: '{png_filepath}'")
    except Exception as e:
        print(f"Error plotting or saving all devices normalized comparison chart: {e}")
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS"
    
    event_detection_params = {
        "pre_event_samples_ratio": 0.1, "min_pre_event_samples": 30,
        "threshold_factor": 5, "min_abs_threshold": 0.15
    }
    normalization_params = {
        "normalization_window_duration": 1.5 # Seconds after event start to find max amplitude for normalization
    }
    common_time_axis_params = {"num_points": 1000}

    if not os.path.isdir(data_directory):
        print(f"Error: Directory '{data_directory}' not found. Please create it and place .xls files inside.")
        exit()

    xls_file_paths = glob.glob(os.path.join(data_directory, '*.xls'))
    if not xls_file_paths:
        print(f"No .xls files found in '{data_directory}'.")
        exit()

    print(f"Found {len(xls_file_paths)} .xls files in '{data_directory}'.")
    
    all_raw_data = {}
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath)
        print(f"  Reading: {filename}...")
        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)
        if time_data and acceleration_data:
            all_raw_data[filename] = {'time': time_data, 'acceleration': acceleration_data}
        else:
            print(f"    Skipping file {filename} due to read error.")
    
    if not all_raw_data:
        print("Failed to read data from any file. Exiting.")
        exit()

    device_groups = defaultdict(list)
    for filename in all_raw_data.keys():
        if len(filename) >= 2:
            device_id = filename[:2].upper()
            device_groups[device_id].append(filename)
        else:
            print(f"Warning: Filename {filename} too short for device ID, ignoring.")

    if not device_groups:
        print("Could not group any files by device ID. Check naming convention ('XXN.xls').")
        exit()
        
    print("\nFiles grouped by device:")
    for device_id, files in device_groups.items():
        print(f"  Device {device_id}: {', '.join(files)}")

    all_device_normalized_average_curves = {}
    
    print("\nCalculating normalized average curve for each device...")
    for device_id, filenames_in_group in device_groups.items():
        print(f"  Processing Device: {device_id} (contains {len(filenames_in_group)} files)")
        aligned_and_normalized_datasets_for_this_device = []
        for filename in filenames_in_group:
            if filename in all_raw_data:
                raw_data = all_raw_data[filename]
                time_orig, accel_orig = raw_data['time'], raw_data['acceleration']
                
                t_start_event, event_idx = find_event_start_time(time_orig, accel_orig, filename, **event_detection_params)
                
                # Normalize acceleration data for THIS SPECIFIC FILE
                accel_normalized = normalize_acceleration_amplitude(time_orig, accel_orig, event_idx, 
                                                                  **normalization_params)
                
                time_adjusted = [t - t_start_event for t in time_orig]
                
                aligned_and_normalized_datasets_for_this_device.append(
                    (time_adjusted, accel_normalized, filename, t_start_event, event_idx)
                )
            else:
                print(f"    Warning: Raw data for file {filename} of device {device_id} not found.")
        
        if aligned_and_normalized_datasets_for_this_device:
            avg_time, avg_norm_accel = calculate_device_average_normalized_curve(
                aligned_and_normalized_datasets_for_this_device, common_time_axis_params
            )
            if avg_time is not None and avg_norm_accel is not None:
                all_device_normalized_average_curves[device_id] = (avg_time, avg_norm_accel)
                print(f"    Device {device_id} normalized average curve calculation complete.")
            else:
                print(f"    Failed to calculate normalized average curve for device {device_id}.")
        else:
            print(f"    Device {device_id} has no valid aligned and normalized data for averaging.")

    if all_device_normalized_average_curves:
        plot_all_devices_normalized_comparison(
            all_device_normalized_average_curves, 
            "all_devices_normalized_merged_comparison.png", 
            output_plots_directory
        )
    else:
        print("\nNo device normalized average curves were calculated. Cannot generate final comparison plot.")

    print("\nProgram processing complete.")