import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np # 用於數學計算，如標準差

def read_xls_columns_to_arrays(filepath):
    """
    讀取 XLS 檔案中指定的 A 列和 C 列，並將其內容存為陣列。
    """
    column_a_name = 'Time (s)'
    column_c_name = 'Acceleration y (m/s^2)'
    filename_for_error = os.path.basename(filepath)

    try:
        df = pd.read_excel(filepath, usecols=[column_a_name, column_c_name], engine='xlrd')
        time_array = df[column_a_name].astype(float).tolist() # 確保時間是浮點數
        acceleration_array = df[column_c_name].astype(float).tolist() # 確保加速度是浮點數
        return time_array, acceleration_array
    except FileNotFoundError:
        print(f"  錯誤：找不到檔案 '{filepath}'")
        return None, None
    except ValueError as e: # 更具體的錯誤捕獲
        print(f"  錯誤：檔案 '{filename_for_error}' 中指定的欄位 '{column_a_name}' 或 '{column_c_name}' 找不到，或數據無法轉換為數字。請檢查欄位名稱和數據內容。")
        print(f"  詳細錯誤: {e}")
        try:
            all_headers_df = pd.read_excel(filepath, nrows=0, engine='xlrd')
            print(f"  檔案 '{filename_for_error}' 中可用的欄位標頭有: {all_headers_df.columns.tolist()}")
        except Exception as e_inner:
            print(f"  嘗試讀取檔案 '{filename_for_error}' 的欄位標頭失敗: {e_inner}")
        return None, None
    except Exception as e:
        print(f"  讀取或處理檔案 '{filename_for_error}' 時發生未預期錯誤：{e}")
        return None, None

def find_event_start_time(time_data, accel_data, filename,
                          pre_event_samples_ratio=0.1, # 用於計算基線的數據點比例
                          min_pre_event_samples=20,    # 最少用於基線的數據點
                          threshold_factor=5,          # 超過基線標準差的倍數
                          min_abs_threshold=0.2):      # 最小絕對加速度閾值
    """
    嘗試自動偵測震動事件的開始時間。
    返回估計的事件開始時間。如果無法偵測，返回 0.0。
    """
    if not time_data or not accel_data or len(time_data) < min_pre_event_samples:
        print(f"  警告 ({filename}): 數據不足以偵測事件開始時間，將使用 0.0 作為開始時間。")
        return 0.0

    num_samples = len(accel_data)
    num_pre_event_samples = max(min_pre_event_samples, int(num_samples * pre_event_samples_ratio))
    num_pre_event_samples = min(num_pre_event_samples, num_samples -1) # 確保不超過總長度

    if num_pre_event_samples <= 0: # 如果數據點太少
        print(f"  警告 ({filename}): 計算基線的樣本數不足，將使用 0.0 作為開始時間。")
        return 0.0

    baseline_accel = accel_data[:num_pre_event_samples]
    baseline_std = np.std(baseline_accel)
    
    # 實際使用的閾值是 (基線標準差*倍數) 和 (最小絕對閾值) 中的較大者
    # 這樣可以避免在基線非常平靜時，微小波動被誤判為事件
    effective_threshold = max(baseline_std * threshold_factor, min_abs_threshold)
    
    # 防止 baseline_std 非常小時，effective_threshold 過小
    if baseline_std < 1e-6 and min_abs_threshold == 0 : # 如果基線幾乎為零且沒有最小絕對閾值
         effective_threshold = 0.05 # 給一個小的預設值，避免誤觸發
    elif effective_threshold < 1e-6: # 如果計算出的閾值過小
        effective_threshold = min_abs_threshold if min_abs_threshold > 0 else 0.05


    print(f"  ({filename}): 基線標準差={baseline_std:.4f}, 有效加速度閾值={effective_threshold:.4f} (基於前 {num_pre_event_samples} 個點)")

    event_start_time = 0.0
    event_detected = False
    # 從基線期之後開始搜索
    for i in range(num_pre_event_samples, num_samples):
        if abs(accel_data[i]) > effective_threshold:
            event_start_time = time_data[i]
            event_detected = True
            print(f"  ({filename}): 偵測到事件開始於時間 {event_start_time:.4f} s (加速度 {accel_data[i]:.4f})")
            break
    
    if not event_detected:
        print(f"  警告 ({filename}): 未能自動偵測到明顯的事件開始點 (使用閾值 {effective_threshold:.4f})。將使用 0.0s 作為相對起點。")
        # 可以考慮返回第一個時間點 time_data[0] 或保持 0.0
        # 如果返回 time_data[0]，則對齊後的起點將是原始數據的起點
        # 如果返回 0.0，則對於沒有偵測到事件的檔案，其時間軸不會平移
        # 這裡選擇返回 0.0，意味著如果沒偵測到，它的時間軸就是原始時間軸
        # 但在繪圖時，我們會從所有偵測到的最小時間開始繪製，或者乾脆讓未偵測到的檔案也減去一個"假設"的起始時間
        # 為了簡化，如果未偵測到，我們將其事件開始時間視為數據的第一個時間點
        if time_data:
            event_start_time = time_data[0] # 使用第一個時間點作為"開始"
            print(f"  ({filename}): 將使用數據的第一個時間點 {event_start_time:.4f}s 作為參考開始時間。")


    return event_start_time


def save_multi_comparison_plot(datasets_to_plot, output_filename_base, output_directory="OUTPUT_PLOTS"):
    """
    繪製多個已對齊的數據集的時間與加速度關係圖，並儲存為 PNG。
    datasets_to_plot: list of tuples, e.g., [(time1_adj, accel1, label1, orig_start1), ...]
    """
    if not datasets_to_plot:
        print("  警告：沒有數據集可供繪製比較圖。")
        return

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"  已建立儲存圖片的目錄: '{output_directory}'")
        except OSError as e:
            print(f"  錯誤：無法建立目錄 '{output_directory}': {e}")
            return

    # 確保輸出檔名不會太長
    if len(output_filename_base) > 100:
        output_filename_base = output_filename_base[:100] + "_etc"
    png_filename = f"comparison_{output_filename_base}.png"
    png_filepath = os.path.join(output_directory, png_filename)

    plt.figure(figsize=(15, 8)) # 較大的圖以便容納多個曲線

    min_rel_time = float('inf')
    max_rel_time = float('-inf')

    for i, (time_adj, accel, label, orig_start_time) in enumerate(datasets_to_plot):
        if time_adj and accel:
            plt.plot(time_adj, accel, marker='.', linestyle='-', markersize=3, label=f"{label} (原始開始: {orig_start_time:.2f}s)", alpha=0.7)
            if time_adj: # 確保列表不為空
                min_rel_time = min(min_rel_time, min(time_adj))
                max_rel_time = max(max_rel_time, max(time_adj))

    plt.xlabel('相對時間 (s) [事件開始點對齊於 t\'=0]')
    plt.ylabel('Acceleration y (m/s^2)')
    plt.title(f'多檔案加速度比較 (事件對齊)')
    plt.grid(True)
    plt.legend(fontsize='small') # 圖例，字體稍小以容納更多條目

    # 根據數據範圍設定合理的X軸範圍，例如從第一個事件點附近開始
    # if min_rel_time != float('inf') and max_rel_time != float('-inf'):
    #     plt.xlim(max(-1, min_rel_time -0.5) , max_rel_time + 0.5) # 稍微擴展顯示範圍

    try:
        plt.savefig(png_filepath, dpi=150) # dpi可以提高圖片解析度
        plt.close()
        print(f"  成功將比較圖表儲存為: '{png_filepath}'")
    except Exception as e:
        print(f"  繪製或儲存比較圖表 '{png_filename}' 時發生錯誤: {e}")
        plt.close()


# --- 主要執行部分 ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS"
    all_files_data = {} # 儲存所有已讀取檔案的數據: {filename: {'time': [], 'acceleration': []}}

    # 偵測事件開始時間的參數 (可以根據需要調整)
    event_detection_params = {
        "pre_event_samples_ratio": 0.1,
        "min_pre_event_samples": 30, # 增加樣本數以獲得更穩定的基線
        "threshold_factor": 5,
        "min_abs_threshold": 0.15   # 根據您圖表的Y軸範圍，這個值可能需要調整
    }


    if not os.path.isdir(data_directory):
        print(f"錯誤：目錄 '{data_directory}' 不存在或不是一個目錄。")
        exit()

    search_pattern = os.path.join(data_directory, '*.xls')
    xls_file_paths = glob.glob(search_pattern)

    if not xls_file_paths:
        print(f"在目錄 '{data_directory}' 中沒有找到任何 .xls 檔案。")
        exit()

    print(f"在 '{data_directory}' 目錄中找到以下 .xls 檔案將進行處理:")
    for fp in xls_file_paths:
        print(f"  - {os.path.basename(fp)}")
    print("-" * 30)

    successfully_processed_files = [] # 儲存成功讀取數據的檔案名稱
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath)
        print(f"\n正在讀取檔案: {filename}...")
        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)

        if time_data is not None and acceleration_data is not None:
            all_files_data[filename] = {
                'time': time_data,
                'acceleration': acceleration_data
            }
            successfully_processed_files.append(filename)
            print(f"  成功讀取檔案 '{filename}'。 (數據點: {len(time_data)})")
        else:
            print(f"  未能從檔案 '{filename}' 讀取所需的數據。")

    print("\n" + "=" * 30)
    if not successfully_processed_files:
        print("沒有成功讀取任何檔案的數據，無法進行後續操作。")
        exit()

    # --- 讓使用者選擇多個檔案進行比較 ---
    print("已成功讀取數據的檔案列表：")
    for i, fname in enumerate(successfully_processed_files):
        print(f"  {i + 1}. {fname}")

    selected_for_comparison_filenames = []
    while True:
        try:
            choices_str = input(f"\n請輸入要比較的多個檔案的名稱或編號 (用逗號分隔，例如 '1,3,RP2.xls'，或輸入 'q' 離開): ").strip()
            if choices_str.lower() == 'q':
                break
            if not choices_str:
                print("未輸入任何選擇。")
                continue

            chosen_items = [item.strip() for item in choices_str.split(',')]
            current_selection_filenames = []
            valid_selection = True
            for item in chosen_items:
                selected_file = None
                if item.isdigit():
                    choice_idx = int(item) - 1
                    if 0 <= choice_idx < len(successfully_processed_files):
                        selected_file = successfully_processed_files[choice_idx]
                    else:
                        print(f"  錯誤：編號 '{item}' 無效。")
                        valid_selection = False; break
                elif item in successfully_processed_files:
                    selected_file = item
                else: # 嘗試加上 .xls 後綴再檢查一次
                    item_with_xls = item if item.endswith(".xls") else item + ".xls"
                    if item_with_xls in successfully_processed_files:
                        selected_file = item_with_xls
                    else:
                        print(f"  錯誤：檔案名稱 '{item}' (或 '{item_with_xls}') 無效或未處理。")
                        valid_selection = False; break
                
                if selected_file and selected_file not in current_selection_filenames: # 避免重複
                    current_selection_filenames.append(selected_file)
            
            if valid_selection and len(current_selection_filenames) >= 2:
                selected_for_comparison_filenames = current_selection_filenames
                break # 選擇有效且至少有兩個檔案，跳出選擇迴圈
            elif valid_selection and len(current_selection_filenames) < 2:
                print("請至少選擇兩個檔案進行比較。")
            else:
                print("選擇無效，請重新輸入。")


        except ValueError:
            print("輸入格式錯誤，請確保編號是數字。")
        except Exception as e:
            print(f"處理選擇時發生錯誤: {e}")

    if not selected_for_comparison_filenames:
        print("未選擇任何檔案進行比較，程式結束。")
        exit()

    print("\n將比較以下檔案：")
    for fname in selected_for_comparison_filenames:
        print(f"  - {fname}")
    
    # --- 對選中的檔案進行時間對齊和繪圖 ---
    datasets_for_plot = []
    output_filename_parts = [] # 用於產生輸出圖片檔名

    print("\n正在對齊數據並準備繪圖...")
    for filename_xls in selected_for_comparison_filenames:
        if filename_xls in all_files_data:
            data = all_files_data[filename_xls]
            time_orig = data['time']
            accel_orig = data['acceleration']

            # 偵測事件開始時間
            t_start_event = find_event_start_time(time_orig, accel_orig, filename_xls, **event_detection_params)
            
            # 時間軸對齊
            time_adjusted = [t - t_start_event for t in time_orig]
            
            datasets_for_plot.append((time_adjusted, accel_orig, filename_xls, t_start_event))
            output_filename_parts.append(os.path.splitext(filename_xls)[0]) # 取檔名部分
        else:
            print(f"警告：檔案 {filename_xls} 的數據未找到，將跳過。")

    if datasets_for_plot:
        comparison_filename_base = "_vs_".join(output_filename_parts)
        save_multi_comparison_plot(datasets_for_plot, comparison_filename_base, output_plots_directory)
    else:
        print("沒有可繪製的數據。")

    print("\n程式結束。")