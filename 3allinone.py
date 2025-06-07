import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib 中文顯示設定 ---
# 嘗試幾種常見的中文字體，選擇一個您的系統上有的
font_names = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 嘗試設置字體
font_found = False
for font_name in font_names:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        # 測試一下字體是否真的可用 (有些系統可能列表裡有但實際無法加載)
        fig_test, ax_test = plt.subplots(figsize=(1,1))
        ax_test.set_title("測試")
        plt.close(fig_test)
        font_found = True
        print(f"成功設置 Matplotlib 字體為: {font_name}")
        break
    except Exception as e:
        print(f"字體 {font_name} 設置失敗或不可用: {e}")

if not font_found:
    print("警告：未能成功設置中文字體，圖表中的中文可能無法正常顯示。請確保您的系統已安裝並配置了 Matplotlib 可用的中文字體。")
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
    except FileNotFoundError:
        print(f"  錯誤：找不到檔案 '{filepath}'")
        return None, None
    except ValueError as e:
        print(f"  錯誤：檔案 '{filename_for_error}' 中欄位 '{column_a_name}' 或 '{column_c_name}' 找不到，或數據無法轉為數字。")
        print(f"  詳細錯誤: {e}")
        try:
            all_headers_df = pd.read_excel(filepath, nrows=0, engine='xlrd')
            print(f"  檔案 '{filename_for_error}' 中可用欄位標頭: {all_headers_df.columns.tolist()}")
        except Exception: pass
        return None, None
    except Exception as e:
        print(f"  讀取或處理檔案 '{filename_for_error}' 時發生錯誤：{e}")
        return None, None

def find_event_start_time(time_data, accel_data, filename,
                          pre_event_samples_ratio=0.1,
                          min_pre_event_samples=30,
                          threshold_factor=5,
                          min_abs_threshold=0.15):
    if not time_data or not accel_data or len(time_data) < min_pre_event_samples:
        print(f"  警告 ({filename}): 數據不足以偵測事件開始，將使用 0.0s。")
        return 0.0

    num_samples = len(accel_data)
    num_pre_event_samples = max(min_pre_event_samples, int(num_samples * pre_event_samples_ratio))
    num_pre_event_samples = min(num_pre_event_samples, num_samples -1)

    if num_pre_event_samples <= 0:
        print(f"  警告 ({filename}): 計算基線樣本數不足，將使用 0.0s。")
        return 0.0

    baseline_accel = np.array(accel_data[:num_pre_event_samples])
    baseline_std = np.std(baseline_accel)
    effective_threshold = max(baseline_std * threshold_factor, min_abs_threshold)
    if baseline_std < 1e-6 and min_abs_threshold == 0 :
         effective_threshold = 0.05
    elif effective_threshold < 1e-6:
        effective_threshold = min_abs_threshold if min_abs_threshold > 0 else 0.05

    print(f"  ({filename}): 基線標準差={baseline_std:.4f}, 有效閾值={effective_threshold:.4f} (基於前 {num_pre_event_samples} 點)")

    event_start_time = time_data[0] # 預設為第一個時間點
    event_detected = False
    for i in range(num_pre_event_samples, num_samples):
        if abs(accel_data[i]) > effective_threshold:
            event_start_time = time_data[i]
            event_detected = True
            print(f"  ({filename}): 偵測到事件開始於 {event_start_time:.4f} s (加速度 {accel_data[i]:.4f})")
            break
    
    if not event_detected:
        print(f"  警告 ({filename}): 未偵測到明顯事件 (閾值 {effective_threshold:.4f})。將使用第一個數據點的時間 {time_data[0]:.4f}s 作為參考開始。")
    return event_start_time


def plot_and_save_merged_data(aligned_datasets, common_time_axis_params, output_filename_base, output_directory="OUTPUT_PLOTS"):
    """
    將多個對齊後的數據集合併（平均），繪製並儲存。
    aligned_datasets: list of tuples [(time_adjusted, accel_orig, filename, orig_start_time), ...]
    common_time_axis_params: dict {'num_points': 1000} (或其他定義共同時間軸的方式)
    """
    if not aligned_datasets or len(aligned_datasets) < 1: # 至少需要一個數據集才能"合併"（即使是單個）
        print("  警告：沒有數據集或數據集數量不足以合併繪圖。")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"  已建立儲存圖片目錄: '{output_directory}'")
        except OSError as e:
            print(f"  錯誤：無法建立目錄 '{output_directory}': {e}")
            return

    # 1. 準備插值所需的數據並確定共同時間軸的範圍
    all_interp_accel = []
    min_rel_time = float('inf')
    max_rel_time = float('-inf')

    for time_adj, accel, _, _ in aligned_datasets:
        if time_adj: # 確保列表不為空
            min_rel_time = min(min_rel_time, min(time_adj))
            max_rel_time = max(max_rel_time, max(time_adj))

    if min_rel_time == float('inf') or max_rel_time == float('-inf') or min_rel_time >= max_rel_time :
        print("  錯誤：無法確定有效的共同時間軸範圍。")
        return

    # 2. 創建共同的相對時間軸
    num_points_common_axis = common_time_axis_params.get('num_points', 1000)
    common_rel_time_axis = np.linspace(min_rel_time, max_rel_time, num_points_common_axis)

    # 3. 對每個數據集進行插值
    for time_adj, accel, filename, _ in aligned_datasets:
        if not time_adj or not accel:
            print(f"  警告：檔案 {filename} 的對齊數據為空，跳過插值。")
            continue
        try:
            # np.interp 需要 xp (原始x軸) 是遞增的
            # 如果原始時間數據不是嚴格遞增，插值可能會出問題或給出警告
            # 確保 time_adj 是排序好的 (通常讀取時就是，但以防萬一)
            # 這裡假設 time_adj 已經是排序好的
            interp_accel = np.interp(common_rel_time_axis, time_adj, accel)
            all_interp_accel.append(interp_accel)
        except Exception as e:
            print(f"  錯誤：對檔案 {filename} 進行插值時失敗: {e}")
            # 如果一個檔案插值失敗，我們可以選擇跳過它或終止
            # 這裡選擇跳過
            continue
            
    if not all_interp_accel:
        print("  錯誤：沒有成功的插值數據可供平均。")
        return

    # 4. 計算平均值和標準差
    all_interp_accel_np = np.array(all_interp_accel)
    mean_accel = np.mean(all_interp_accel_np, axis=0)
    std_accel = np.std(all_interp_accel_np, axis=0)

    # 5. 繪圖
    png_filename = f"merged_{output_filename_base}.png"
    png_filepath = os.path.join(output_directory, png_filename)

    plt.figure(figsize=(12, 7))
    
    # 繪製平均曲線
    plt.plot(common_rel_time_axis, mean_accel, color='blue', linestyle='-', linewidth=2, label=f'平均加速度 ({len(all_interp_accel)} 次測量)')
    
    # 繪製標準差範圍 (平均 ± 1個標準差)
    plt.fill_between(common_rel_time_axis, mean_accel - std_accel, mean_accel + std_accel,
                     color='lightblue', alpha=0.5, label='平均 ± 1 標準差')

    plt.xlabel('相對時間 (s) [事件開始點對齊於 t\'=0]')
    plt.ylabel('平均加速度 y (m/s^2)')
    plt.title(f'合併 {len(all_interp_accel)} 次測量的平均加速度\n(檔案: {output_filename_base})')
    plt.grid(True)
    plt.legend(fontsize='medium')
    
    # 可以根據數據調整Y軸範圍
    y_min = np.min(mean_accel - std_accel)
    y_max = np.max(mean_accel + std_accel)
    padding = (y_max - y_min) * 0.1 # 10% 的上下邊距
    plt.ylim(y_min - padding, y_max + padding)


    try:
        plt.savefig(png_filepath, dpi=150)
        plt.close()
        print(f"  成功將合併圖表儲存為: '{png_filepath}'")
    except Exception as e:
        print(f"  繪製或儲存合併圖表 '{png_filename}' 時發生錯誤: {e}")
        plt.close()


# --- 主要執行部分 ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS"
    all_files_data = {}

    event_detection_params = {
        "pre_event_samples_ratio": 0.1, "min_pre_event_samples": 30,
        "threshold_factor": 5, "min_abs_threshold": 0.15
    }
    common_time_axis_params = {"num_points": 1000} # 用於插值的共同時間軸的點數


    if not os.path.isdir(data_directory):
        print(f"錯誤：目錄 '{data_directory}' 不存在。")
        exit()

    xls_file_paths = glob.glob(os.path.join(data_directory, '*.xls'))
    if not xls_file_paths:
        print(f"在 '{data_directory}' 中未找到 .xls 檔案。")
        exit()

    print(f"在 '{data_directory}' 找到以下 .xls 檔案:")
    for fp in xls_file_paths: print(f"  - {os.path.basename(fp)}")
    print("-" * 30)

    successfully_processed_files = []
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath)
        print(f"\n正在讀取檔案: {filename}...")
        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)
        if time_data and acceleration_data:
            all_files_data[filename] = {'time': time_data, 'acceleration': acceleration_data}
            successfully_processed_files.append(filename)
            print(f"  成功讀取檔案 '{filename}'。 (數據點: {len(time_data)})")
        else:
            print(f"  未能從檔案 '{filename}' 讀取數據。")

    print("\n" + "=" * 30)
    if not successfully_processed_files:
        print("未成功讀取任何檔案數據。")
        exit()

    print("已成功讀取數據的檔案列表：")
    for i, fname in enumerate(successfully_processed_files): print(f"  {i + 1}. {fname}")

    selected_for_operation_filenames = []
    while True:
        try:
            choices_str = input(f"\n請選擇要操作的檔案 (至少1個，用逗號分隔，例如 '1,3,RP2.xls'，或輸入 'q' 離開): ").strip()
            if choices_str.lower() == 'q': break
            if not choices_str: continue

            chosen_items = [item.strip() for item in choices_str.split(',')]
            current_selection_filenames = []
            valid_selection = True
            for item in chosen_items:
                selected_file = None
                if item.isdigit():
                    choice_idx = int(item) - 1
                    if 0 <= choice_idx < len(successfully_processed_files):
                        selected_file = successfully_processed_files[choice_idx]
                    else: valid_selection = False; print(f"  錯誤：編號 '{item}' 無效。"); break
                elif item in successfully_processed_files: selected_file = item
                else:
                    item_with_xls = item if item.endswith(".xls") else item + ".xls"
                    if item_with_xls in successfully_processed_files: selected_file = item_with_xls
                    else: valid_selection = False; print(f"  錯誤：檔案 '{item}' 無效。"); break
                if selected_file and selected_file not in current_selection_filenames:
                    current_selection_filenames.append(selected_file)
            
            if valid_selection and len(current_selection_filenames) >= 1: # 至少一個檔案
                selected_for_operation_filenames = current_selection_filenames
                break
            elif valid_selection : print("請至少選擇一個檔案。")
            else: print("選擇無效，請重新輸入。")
        except ValueError: print("輸入格式錯誤。")
        except Exception as e: print(f"處理選擇時發生錯誤: {e}")

    if not selected_for_operation_filenames:
        print("未選擇檔案，程式結束。")
        exit()

    print("\n將操作以下檔案：")
    for fname in selected_for_operation_filenames: print(f"  - {fname}")
    
    aligned_datasets_for_plot = []
    output_filename_parts = []

    print("\n正在對齊數據...")
    for filename_xls in selected_for_operation_filenames:
        if filename_xls in all_files_data:
            data = all_files_data[filename_xls]
            time_orig, accel_orig = data['time'], data['acceleration']
            t_start_event = find_event_start_time(time_orig, accel_orig, filename_xls, **event_detection_params)
            time_adjusted = [t - t_start_event for t in time_orig]
            aligned_datasets_for_plot.append((time_adjusted, accel_orig, filename_xls, t_start_event))
            output_filename_parts.append(os.path.splitext(filename_xls)[0])
        else:
            print(f"警告：檔案 {filename_xls} 數據未找到，跳過。")

    if aligned_datasets_for_plot:
        # 如果只有一個檔案，也可以用這個函數繪製（它不會顯示標準差陰影，只有一條線）
        # 或者你可以為單個檔案保留之前的單獨繪圖函數邏輯
        # 這裡我們統一使用合併函數
        merged_filename_base = "_".join(output_filename_parts)
        if len(selected_for_operation_filenames) == 1:
            merged_filename_base = output_filename_parts[0] # 單個檔案名
            print(f"\n選擇了單個檔案 '{selected_for_operation_filenames[0]}', 將繪製其對齊後的圖表 (無平均)。")
        else:
             print(f"\n選擇了 {len(selected_for_operation_filenames)} 個檔案, 將繪製它們的合併平均圖表。")

        plot_and_save_merged_data(aligned_datasets_for_plot, common_time_axis_params, merged_filename_base, output_plots_directory)
    else:
        print("沒有可繪製的數據。")

    print("\n程式結束。")