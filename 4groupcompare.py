import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict # 用於方便地分組

# --- Matplotlib 中文顯示設定 ---
font_names = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
font_found = False
for font_name in font_names:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        fig_test, ax_test = plt.subplots(figsize=(0.1,0.1)); ax_test.set_title("測"); plt.close(fig_test) # 極小的測試圖
        font_found = True
        print(f"成功設置 Matplotlib 字體為: {font_name}")
        break
    except Exception:
        pass
if not font_found:
    print("警告：未能成功設置中文字體，圖表中的中文可能無法正常顯示。")
# ---------------------------------

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
        print(f"  錯誤 (檔案: {filename_for_error}): 讀取或轉換數據失敗。檢查欄位名及內容。詳細: {e}")
        return None, None

def find_event_start_time(time_data, accel_data, filename,
                          pre_event_samples_ratio=0.1, min_pre_event_samples=30,
                          threshold_factor=5, min_abs_threshold=0.15):
    if not time_data or not accel_data or len(time_data) < min_pre_event_samples:
        print(f"  警告 ({filename}): 數據不足以偵測事件，使用 0.0s。")
        return 0.0
    num_samples = len(accel_data)
    num_pre_event_samples = min(max(min_pre_event_samples, int(num_samples * pre_event_samples_ratio)), num_samples - 1)
    if num_pre_event_samples <= 0: return 0.0

    baseline_accel = np.array(accel_data[:num_pre_event_samples])
    baseline_std = np.std(baseline_accel)
    effective_threshold = max(baseline_std * threshold_factor, min_abs_threshold)
    if baseline_std < 1e-6 and min_abs_threshold == 0: effective_threshold = 0.05
    elif effective_threshold < 1e-6: effective_threshold = min_abs_threshold if min_abs_threshold > 0 else 0.05
    
    # print(f"  ({filename}): 基線標準差={baseline_std:.4f}, 有效閾值={effective_threshold:.4f}")
    event_start_time = time_data[0]
    for i in range(num_pre_event_samples, num_samples):
        if abs(accel_data[i]) > effective_threshold:
            event_start_time = time_data[i]
            # print(f"  ({filename}): 偵測到事件開始於 {event_start_time:.4f}s")
            return event_start_time
    # print(f"  警告 ({filename}): 未偵測到明顯事件，使用第一個數據點的時間 {time_data[0]:.4f}s。")
    return event_start_time

def calculate_device_average_curve(aligned_datasets_for_device, common_time_axis_params):
    """
    計算單個裝置（一組已對齊的實驗數據）的平均曲線。
    返回 (common_rel_time_axis, mean_accel) 或 (None, None) 如果失敗。
    """
    if not aligned_datasets_for_device: return None, None

    all_interp_accel = []
    min_rel_time_device = float('inf')
    max_rel_time_device = float('-inf')

    for time_adj, _, _, _ in aligned_datasets_for_device:
        if time_adj:
            min_rel_time_device = min(min_rel_time_device, min(time_adj))
            max_rel_time_device = max(max_rel_time_device, max(time_adj))

    if min_rel_time_device == float('inf') or max_rel_time_device == float('-inf') or min_rel_time_device >= max_rel_time_device:
        print(f"  錯誤：裝置數據無法確定有效的共同時間軸範圍。")
        return None, None

    num_points_common_axis = common_time_axis_params.get('num_points', 1000)
    common_rel_time_axis = np.linspace(min_rel_time_device, max_rel_time_device, num_points_common_axis)

    for time_adj, accel, filename, _ in aligned_datasets_for_device:
        if not time_adj or not accel: continue
        try:
            interp_accel = np.interp(common_rel_time_axis, time_adj, accel)
            all_interp_accel.append(interp_accel)
        except Exception as e:
            print(f"  錯誤：對檔案 {filename} 進行裝置內插值時失敗: {e}")
            continue
            
    if not all_interp_accel: return None, None

    mean_accel = np.mean(np.array(all_interp_accel), axis=0)
    return common_rel_time_axis, mean_accel

def plot_all_devices_comparison(all_device_averages, output_filename, output_directory="OUTPUT_PLOTS"):
    """
    將所有裝置的平均曲線繪製在同一張圖上。
    all_device_averages: dict {'DEVICE_ID': (time_axis, mean_accel_curve), ...}
    """
    if not all_device_averages:
        print("沒有裝置平均數據可供繪製。")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except OSError as e:
            print(f"錯誤：無法建立目錄 '{output_directory}': {e}")
            return

    png_filepath = os.path.join(output_directory, output_filename)
    plt.figure(figsize=(14, 8))

    # 取得所有繪圖中 X 軸的最大和最小值，以統一 X 軸範圍
    global_min_x = float('inf')
    global_max_x = float('-inf')

    for device_id, (time_axis, _) in all_device_averages.items():
        if time_axis is not None and len(time_axis) > 0:
            global_min_x = min(global_min_x, np.min(time_axis))
            global_max_x = max(global_max_x, np.max(time_axis))


    for device_id, (time_axis, mean_accel_curve) in all_device_averages.items():
        if time_axis is not None and mean_accel_curve is not None:
            plt.plot(time_axis, mean_accel_curve, marker='.', linestyle='-', markersize=3, linewidth=1.5, label=f'setup {device_id}')
        else:
            print(f"警告：裝置 {device_id} 的平均數據缺失，無法繪製。")

    plt.xlabel('time]')
    plt.ylabel('average acc y (m/s^2)')
    plt.title('acc comparision')
    plt.grid(True)
    plt.legend(fontsize='medium')

    if global_min_x != float('inf') and global_max_x != float('-inf'):
         plt.xlim(global_min_x, global_max_x) # 統一 X 軸

    try:
        plt.savefig(png_filepath, dpi=200) # 提高解析度
        plt.close()
        print(f"成功將所有裝置比較圖表儲存為: '{png_filepath}'")
    except Exception as e:
        print(f"繪製或儲存所有裝置比較圖表時發生錯誤: {e}")
        plt.close()


# --- 主要執行部分 ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS"
    
    # 參數設定
    event_detection_params = {
        "pre_event_samples_ratio": 0.1, "min_pre_event_samples": 30,
        "threshold_factor": 5, "min_abs_threshold": 0.15 # 根據實際數據調整
    }
    common_time_axis_params = {"num_points": 1000} # 每個裝置平均曲線的插值點數

    # 1. 檢查 DATA 目錄
    if not os.path.isdir(data_directory):
        print(f"錯誤：目錄 '{data_directory}' 不存在。請建立該目錄並放入 .xls 檔案。")
        exit()

    # 2. 讀取所有 XLS 檔案
    xls_file_paths = glob.glob(os.path.join(data_directory, '*.xls'))
    if not xls_file_paths:
        print(f"在 '{data_directory}' 中未找到任何 .xls 檔案。")
        exit()

    print(f"在 '{data_directory}' 目錄中找到 {len(xls_file_paths)} 個 .xls 檔案。")
    
    all_raw_data = {} # {filename: {'time': [], 'acceleration': []}}
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath)
        print(f"  正在讀取: {filename}...")
        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)
        if time_data and acceleration_data:
            all_raw_data[filename] = {'time': time_data, 'acceleration': acceleration_data}
        else:
            print(f"    跳過檔案 {filename} 因讀取失敗。")
    
    if not all_raw_data:
        print("未能成功讀取任何檔案的數據。程式結束。")
        exit()

    # 3. 按裝置ID分組檔案
    device_groups = defaultdict(list)
    for filename in all_raw_data.keys():
        if len(filename) >= 2: # 確保檔名至少有2個字元作為ID
            device_id = filename[:2].upper() # 取前兩個字母作為ID，並轉大寫
            device_groups[device_id].append(filename)
        else:
            print(f"警告：檔案 {filename} 名稱過短，無法確定裝置ID，將被忽略。")

    if not device_groups:
        print("未能根據檔名規則分組任何檔案。請檢查檔名是否符合 'XXN.xls' 格式。")
        exit()
        
    print("\n檔案已按裝置分組:")
    for device_id, files in device_groups.items():
        print(f"  裝置 {device_id}: {', '.join(files)}")

    # 4. 為每個裝置組計算平均曲線
    all_device_average_curves = {} # {'DEVICE_ID': (time_axis, mean_accel_curve)}
    
    print("\n正在計算每個裝置的平均曲線...")
    for device_id, filenames_in_group in device_groups.items():
        print(f"  處理裝置: {device_id} (包含 {len(filenames_in_group)} 個檔案)")
        aligned_datasets_for_this_device = []
        for filename in filenames_in_group:
            if filename in all_raw_data:
                raw_data = all_raw_data[filename]
                time_orig, accel_orig = raw_data['time'], raw_data['acceleration']
                
                t_start_event = find_event_start_time(time_orig, accel_orig, filename, **event_detection_params)
                time_adjusted = [t - t_start_event for t in time_orig]
                
                aligned_datasets_for_this_device.append((time_adjusted, accel_orig, filename, t_start_event))
            else: # 理論上不應發生，因為 device_groups 是從 all_raw_data 的鍵生成的
                print(f"    警告：裝置 {device_id} 的檔案 {filename} 的原始數據未找到。")
        
        if aligned_datasets_for_this_device:
            avg_time, avg_accel = calculate_device_average_curve(aligned_datasets_for_this_device, common_time_axis_params)
            if avg_time is not None and avg_accel is not None:
                all_device_average_curves[device_id] = (avg_time, avg_accel)
                print(f"    裝置 {device_id} 平均曲線計算完成。")
            else:
                print(f"    未能計算裝置 {device_id} 的平均曲線。")
        else:
            print(f"    裝置 {device_id} 沒有有效的對齊數據進行平均。")

    # 5. 繪製所有裝置平均曲線的比較圖
    if all_device_average_curves:
        plot_all_devices_comparison(all_device_average_curves, "all_devices_merged_comparison.png", output_plots_directory)
    else:
        print("\n沒有計算出任何裝置的平均曲線，無法生成最終比較圖。")

    print("\n程式處理完成。")