import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import scipy.signal # Import scipy.signal

# Matplotlib setup
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
        print(f"   Error (File: {filename_for_error}): Failed to read or convert data. Details: {e}")
        return None, None

def find_event_start_time(time_data, accel_data, filename,
                             pre_event_samples_ratio=0.1, min_pre_event_samples=30,
                             threshold_factor=5, min_abs_threshold=0.15):
    if not time_data or not accel_data or len(time_data) < min_pre_event_samples:
        return 0.0, 0
    num_samples = len(accel_data)
    num_pre_event_samples = min(max(min_pre_event_samples, int(num_samples * pre_event_samples_ratio)), num_samples - 1)
    if num_pre_event_samples <= 0: return 0.0, 0

    baseline_accel = np.array(accel_data[:num_pre_event_samples])
    baseline_std = np.std(baseline_accel)
    effective_threshold = max(baseline_std * threshold_factor, min_abs_threshold)
    if baseline_std < 1e-6 and min_abs_threshold == 0: effective_threshold = 0.05
    elif effective_threshold < 1e-6: effective_threshold = min_abs_threshold if min_abs_threshold > 0 else 0.05
    
    event_start_time = time_data[0]
    event_start_index = 0
    for i in range(num_pre_event_samples, num_samples):
        if abs(accel_data[i]) > effective_threshold:
            event_start_time = time_data[i]
            event_start_index = i
            return event_start_time, event_start_index
    return event_start_time, event_start_index

def normalize_acceleration_amplitude(time_data, accel_data, event_start_index, normalization_window_duration=1.0):
    if not accel_data: return []
    if event_start_index >= len(time_data): return accel_data

    window_end_time = time_data[event_start_index] + normalization_window_duration
    normalization_window_accel_values = []
    for i in range(event_start_index, len(time_data)):
        if time_data[i] <= window_end_time:
            normalization_window_accel_values.append(accel_data[i])
        else:
            break
            
    if not normalization_window_accel_values:
        fallback_points = min(10, len(accel_data) - event_start_index)
        if fallback_points > 0:
            normalization_window_accel_values = accel_data[event_start_index : event_start_index + fallback_points]
        else:
             if accel_data:
                max_abs_accel_for_norm = np.max(np.abs(accel_data))
                if max_abs_accel_for_norm > 1e-6:
                    return (np.array(accel_data) / max_abs_accel_for_norm).tolist()
             return accel_data

    if not normalization_window_accel_values: return accel_data
    max_abs_accel_for_norm = np.max(np.abs(normalization_window_accel_values))
    if max_abs_accel_for_norm < 1e-6: return accel_data
    return (np.array(accel_data) / max_abs_accel_for_norm).tolist()

# 核心修改：只提取正向峰值
def extract_positive_peak_envelope_data(time_data, accel_data, prominence_factor=0.3):
    """
    從時間和加速度數據中提取正向波峰（局部最大值）的包絡線。
    prominence_factor: 峰值顯著性，作為最大振幅的比例，用於過濾噪音。
    """
    if not time_data or not accel_data:
        return [], []

    accel_np = np.array(accel_data)
    
    # 計算基於最大絕對振幅的 prominence
    max_accel_abs = np.max(np.abs(accel_np))
    if max_accel_abs < 1e-6:
        return [], []

    prominence_val = max_accel_abs * prominence_factor
    print(f"      Envelope prominence threshold for positive peaks: {prominence_val:.4f}") 
    
    # 只尋找正向峰值 (值必須大於0，且顯著性足夠)
    # 增加 height 參數，確保只考慮正值峰
    peaks_pos, _ = scipy.signal.find_peaks(accel_np, height=0, prominence=prominence_val)

    if len(peaks_pos) < 2: # 至少需要兩個峰值才能形成一條線
        print("      Not enough significant positive peaks found for envelope. Returning empty data.")
        return [], []

    envelope_time = [time_data[i] for i in peaks_pos]
    envelope_accel = [accel_np[i] for i in peaks_pos] # 直接取正向振幅

    return envelope_time, envelope_accel


def calculate_device_average_normalized_curve(device_id, filenames_in_group, all_raw_data, 
                                              event_detection_params, normalization_params, 
                                              common_time_axis_params, calculate_envelope=False,
                                              envelope_prominence_factor=0.3): # 傳遞 prominence_factor
    """
    Processes a single device group: aligns, normalizes, and averages.
    Returns (time_axis, mean_normalized_accel) or (None, None).
    如果 calculate_envelope 為 True，則返回 (time_axis, mean_envelope_accel)
    """
    print(f"\n   Processing Device Group: {device_id} (contains {len(filenames_in_group)} files)")
    aligned_and_normalized_datasets = []
    for filename in filenames_in_group:
        if filename in all_raw_data:
            raw_data = all_raw_data[filename]
            time_orig, accel_orig = raw_data['time'], raw_data['acceleration']
            
            t_start_event, event_idx = find_event_start_time(time_orig, accel_orig, filename, **event_detection_params)
            accel_normalized = normalize_acceleration_amplitude(time_orig, accel_orig, event_idx, **normalization_params)
            time_adjusted = [t - t_start_event for t in time_orig]
            
            aligned_and_normalized_datasets.append(
                (time_adjusted, accel_normalized, filename, t_start_event, event_idx)
            )
        else:
            print(f"      Warning: Raw data for file {filename} of device {device_id} not found.")

    if not aligned_and_normalized_datasets:
        print(f"      Device {device_id} has no valid data for averaging.")
        return None, None

    all_interp_data = []
    min_rel_time_device = float('inf')
    max_rel_time_device = float('-inf')

    # 先找到所有數據的相對時間範圍
    for time_adj, _, _, _, _ in aligned_and_normalized_datasets:
        if time_adj:
            min_rel_time_device = min(min_rel_time_device, np.min(time_adj))
            max_rel_time_device = max(max_rel_time_device, np.max(time_adj))

    if min_rel_time_device == float('inf') or max_rel_time_device == float('-inf') or min_rel_time_device >= max_rel_time_device:
        print(f"   Error: Device {device_id} data cannot determine a valid common time axis range for averaging.")
        return None, None

    num_points_common_axis = common_time_axis_params.get('num_points', 1000)
    common_rel_time_axis = np.linspace(min_rel_time_device, max_rel_time_device, num_points_common_axis)

    for time_adj, norm_accel, filename, _, _ in aligned_and_normalized_datasets:
        if not time_adj or not norm_accel: continue
        
        if calculate_envelope:
            # 呼叫新的只提取正向峰值的函數
            envelope_time_single, envelope_accel_single = extract_positive_peak_envelope_data(
                time_adj, norm_accel, prominence_factor=envelope_prominence_factor
            )
            if not envelope_time_single: # 如果無法提取包絡線，跳過此檔案
                print(f"   Warning: Could not extract envelope for file {filename}.")
                continue
            # 對包絡線數據進行插值
            try:
                # 確保包絡線數據的長度足夠，否則 np.interp 可能會出錯
                if len(envelope_time_single) < 2:
                    print(f"   Warning: Not enough points for envelope interpolation for file {filename}. Skipping.")
                    continue
                interp_data = np.interp(common_rel_time_axis, envelope_time_single, envelope_accel_single)
                all_interp_data.append(interp_data)
            except Exception as e:
                print(f"   Error: Envelope interpolation failed for file {filename} (Device {device_id}): {e}")
                continue
        else:
            # 否則，直接對原始的歸一化加速度數據進行插值
            try:
                interp_data = np.interp(common_rel_time_axis, time_adj, norm_accel)
                all_interp_data.append(interp_data)
            except Exception as e:
                print(f"   Error: Interpolation failed for file {filename} (Device {device_id}): {e}")
                continue
            
    if not all_interp_data: 
        print(f"   Error: No successful interpolations for Device {device_id}.")
        return None, None

    mean_curve = np.mean(np.array(all_interp_data), axis=0)
    print(f"      Device {device_id} normalized average curve calculation complete.")
    return common_rel_time_axis, mean_curve


def plot_two_device_comparison(device1_id, device1_data, device2_id, device2_data, 
                                 output_filename_base, output_directory="OUTPUT_PLOTS",
                                 shift_device2_by_n=0.0, plot_envelope=False): 
    if device1_data is None and device2_data is None:
        print("No data available for either selected device to plot.")
        return

    if not os.path.exists(output_directory):
        try: os.makedirs(output_directory)
        except OSError as e: print(f"Error: Cannot create directory '{output_directory}': {e}"); return

    output_filename = f"comparison_{output_filename_base}.png"
    png_filepath = os.path.join(output_directory, output_filename)
    plt.figure(figsize=(14, 8))
    
    global_min_x = float('inf')
    global_max_x = float('-inf')

    if device1_data and device1_data[0] is not None:
        global_min_x = min(global_min_x, np.min(device1_data[0]))
        global_max_x = max(global_max_x, np.max(device1_data[0]))
    
    if device2_data and device2_data[0] is not None:
        shifted_time_axis2 = np.array(device2_data[0]) + shift_device2_by_n
        global_min_x = min(global_min_x, np.min(shifted_time_axis2))
        global_max_x = max(global_max_x, np.max(shifted_time_axis2))

    # --- 裝置1的繪圖 ---
    if device1_data and device1_data[0] is not None and device1_data[1] is not None:
        time_axis1, curve1 = device1_data
        plt.plot(time_axis1, curve1, 
                 linestyle='-', 
                 linewidth=1.5, 
                 label=f'Device {device1_id}')
    else:
        print(f"Warning: Data for Device {device1_id} is incomplete or missing, will not be plotted.")

    # --- 裝置2的繪圖 (新增位移) ---
    if device2_data and device2_data[0] is not None and device2_data[1] is not None:
        time_axis2_orig, curve2 = device2_data
        time_axis2_shifted = np.array(time_axis2_orig) + shift_device2_by_n
        plt.plot(time_axis2_shifted, curve2, 
                 linestyle='-', 
                 linewidth=1.5, 
                 label=f'Device {device2_id} (shifted by {shift_device2_by_n:.2f}s)') 
    else:
        print(f"Warning: Data for Device {device2_id} is incomplete or missing, will not be plotted.")

    # 根據是否繪製包絡線調整 Y 軸標籤和標題
    if plot_envelope:
        plt.ylabel('Normalized Mean Acceleration y (Positive Peak Amplitude)')
        plt.title(f'Comparison: Device {device1_id} vs. Device {device2_id} (Normalized Mean Positive Peak Accel.)')
    else:
        plt.ylabel('Normalized Mean Acceleration y (Amplitude)')
        plt.title(f'Comparison: Device {device1_id} vs. Device {device2_id} (Normalized Mean Accel.)')


    plt.xlabel('Relative Time (s) [Event Start Aligned at t\'=0]')
    plt.grid(True)
    plt.legend(fontsize='medium')
    
    if global_min_x != float('inf') and global_max_x != float('-inf'):
          plt.xlim(global_min_x, global_max_x)
    
    # 調整 Y 軸範圍，因為現在只繪製正向峰值，振幅應該都是正數
    # 但考慮到歸一化，還是可以保留 -1.2 到 1.2
    plt.ylim(-0.2, 1.2) # 讓 Y 軸從接近0開始，更符合正向峰值

    try:
        plt.savefig(png_filepath, dpi=200)
        plt.close()
        print(f"\nSuccessfully saved comparison chart to: '{png_filepath}'")
    except Exception as e:
        print(f"Error plotting or saving comparison chart: {e}")
        plt.close()
# --- Main Execution ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS"
    
    event_detection_params = {
        "pre_event_samples_ratio": 0.1, "min_pre_event_samples": 30,
        "threshold_factor": 5, "min_abs_threshold": 0.15
    }
    normalization_params = {"normalization_window_duration": 1.5}
    common_time_axis_params = {"num_points": 1000}

    if not os.path.isdir(data_directory):
        print(f"Error: Directory '{data_directory}' not found."); exit()

    xls_file_paths = glob.glob(os.path.join(data_directory, '*.xls'))
    if not xls_file_paths:
        print(f"No .xls files found in '{data_directory}'."); exit()

    print(f"Found {len(xls_file_paths)} .xls files. Reading data...")
    all_raw_data = {}
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath)
        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)
        if time_data and acceleration_data:
            all_raw_data[filename] = {'time': time_data, 'acceleration': acceleration_data}
    
    if not all_raw_data: print("Failed to read data from any file. Exiting."); exit()

    device_groups = defaultdict(list)
    for filename in all_raw_data.keys():
        if len(filename) >= 2: device_groups[filename[:2].upper()].append(filename)
    
    if not device_groups: print("Could not group files by device ID."); exit()
        
    available_prefixes = sorted(list(device_groups.keys()))
    print("\nAvailable device prefixes for comparison:")
    for prefix in available_prefixes:
        print(f"   - {prefix} ({len(device_groups[prefix])} files)")

    # --- User Input for two prefixes ---
    selected_prefix1, selected_prefix2 = None, None
    while selected_prefix1 is None:
        user_input1 = input("Enter the FIRST device prefix to compare (e.g., RA): ").strip().upper()
        if user_input1 in available_prefixes:
            selected_prefix1 = user_input1
        else:
            print(f"Invalid prefix '{user_input1}'. Please choose from the list above.")
    
    while selected_prefix2 is None:
        user_input2 = input(f"Enter the SECOND device prefix to compare (cannot be {selected_prefix1}): ").strip().upper()
        if user_input2 == selected_prefix1:
            print("Cannot compare a device with itself. Please choose a different prefix.")
        elif user_input2 in available_prefixes:
            selected_prefix2 = user_input2
        else:
            print(f"Invalid prefix '{user_input2}'. Please choose from the list above.")

    # --- User Input for shift amount ---
    shift_amount = 0.0
    while True:
        try:
            shift_input = input(f"Enter the amount to shift Device {selected_prefix2} curve along the x-axis (e.g., 0.5 for 0.5s to the right, or 0 for no shift): ").strip()
            shift_amount = float(shift_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for the shift.")

    # --- User Input for plotting envelope or raw oscillation ---
    plot_envelope_choice = input("Do you want to plot the positive peak envelope curve (y/n)? ").strip().lower()
    plot_envelope_mode = (plot_envelope_choice == 'y')

    # 如果選擇繪製包絡線，讓用戶輸入 prominence_factor
    envelope_prom_factor = 0.3 # 預設值
    if plot_envelope_mode:
        while True:
            try:
                prom_input = input(f"Enter prominence factor for envelope (e.g., 0.3, higher means fewer peaks): (default {envelope_prom_factor}) ").strip()
                if prom_input:
                    envelope_prom_factor = float(prom_input)
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value for prominence factor.")

    print(f"\nComparing Device {selected_prefix1} with Device {selected_prefix2}")

    # --- Process and get average curves for the two selected devices ---
    device1_avg_data = calculate_device_average_normalized_curve(
        selected_prefix1, device_groups[selected_prefix1], all_raw_data,
        event_detection_params, normalization_params, common_time_axis_params,
        calculate_envelope=plot_envelope_mode,
        envelope_prominence_factor=envelope_prom_factor # 傳遞 prominence_factor
    )
    
    device2_avg_data = calculate_device_average_normalized_curve(
        selected_prefix2, device_groups[selected_prefix2], all_raw_data,
        event_detection_params, normalization_params, common_time_axis_params,
        calculate_envelope=plot_envelope_mode,
        envelope_prominence_factor=envelope_prom_factor # 傳遞 prominence_factor
    )

    # --- Plot the comparison ---
    if device1_avg_data or device2_avg_data: # Proceed if at least one device has data
        output_filename_base = f"{selected_prefix1}_vs_{selected_prefix2}"
        if plot_envelope_mode:
            output_filename_base += f"_PosPeakEnvelope_P{envelope_prom_factor:.2f}"
        output_filename_base += f"_shifted_by_{shift_amount:.2f}s"

        plot_two_device_comparison(
            selected_prefix1, device1_avg_data,
            selected_prefix2, device2_avg_data,
            output_filename_base,
            output_plots_directory,
            shift_device2_by_n=shift_amount,
            plot_envelope=plot_envelope_mode
        )
    else:
        print("\nCould not calculate average curves for either selected device. No comparison plot generated.")

    print("\nProgram processing complete.")