import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def read_xls_columns_to_arrays(filepath):
    """
    讀取 XLS 檔案中指定的 A 列和 C 列，並將其內容存為陣列。

    Args:
        filepath (str): XLS 檔案的路徑。

    Returns:
        tuple: (time_array, acceleration_array)
               如果成功，則返回兩個列表，分別包含時間和加速度數據。
               如果失敗，則返回 (None, None)。
    """
    column_a_name = 'Time (s)'
    column_c_name = 'Acceleration y (m/s^2)'

    try:
        df = pd.read_excel(filepath, usecols=[column_a_name, column_c_name], engine='xlrd')
        time_array = df[column_a_name].tolist()
        acceleration_array = df[column_c_name].tolist()
        return time_array, acceleration_array
    except FileNotFoundError:
        print(f"  錯誤：找不到檔案 '{filepath}'")
        return None, None
    except ValueError as e:
        print(f"  錯誤：檔案 '{os.path.basename(filepath)}' 中指定的欄位找不到。請檢查欄位名稱。")
        print(f"  期望的欄位: ['{column_a_name}', '{column_c_name}']")
        try:
            all_headers_df = pd.read_excel(filepath, nrows=0, engine='xlrd')
            print(f"  檔案中可用的欄位標頭有: {all_headers_df.columns.tolist()}")
        except Exception as e_inner:
            print(f"  嘗試讀取欄位標頭失敗: {e_inner}")
        return None, None
    except Exception as e:
        print(f"  讀取或處理檔案 '{os.path.basename(filepath)}' 時發生未預期錯誤：{e}")
        return None, None

def save_time_acceleration_plot(time_data, acceleration_data, original_filename, output_directory="OUTPUT_PLOTS"):
    """
    繪製時間與加速度的關係圖，並將其儲存為 PNG 檔案。

    Args:
        time_data (list): 時間數據列表。
        acceleration_data (list): 加速度數據列表。
        original_filename (str): 原始 XLS 檔案名稱，用於產生 PNG 檔案名和圖表標題。
        output_directory (str): 儲存 PNG 圖片的目錄名稱。
    """
    if not time_data or not acceleration_data:
        print(f"  警告：檔案 '{original_filename}' 的時間或加速度數據為空，無法產生圖片。")
        return
    if len(time_data) != len(acceleration_data):
        print(f"  警告：檔案 '{original_filename}' 的時間和加速度數據長度不一致 ({len(time_data)} vs {len(acceleration_data)})，無法產生圖片。")
        return

    # 確保輸出目錄存在
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"  已建立儲存圖片的目錄: '{output_directory}'")
        except OSError as e:
            print(f"  錯誤：無法建立目錄 '{output_directory}': {e}")
            return

    # 產生 PNG 檔案名稱 (從原始檔名去除副檔名，加上 .png)
    base_name = os.path.splitext(original_filename)[0]
    png_filename = f"{base_name}.png"
    png_filepath = os.path.join(output_directory, png_filename)

    try:
        plt.figure(figsize=(10, 6)) # 設定圖表大小
        plt.plot(time_data, acceleration_data, marker='.', linestyle='-', markersize=4)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration y (m/s^2)')
        plt.title(f'Time vs. Acceleration for {original_filename}')
        plt.grid(True)

        plt.savefig(png_filepath) # 儲存圖表到檔案
        plt.close() # 關閉圖表，釋放記憶體

        print(f"  成功將圖表儲存為: '{png_filepath}'")

    except Exception as e:
        print(f"  繪製或儲存檔案 '{original_filename}' 的圖表時發生錯誤: {e}")
        # 如果出錯，也嘗試關閉可能未完全處理的 figure
        plt.close()


# --- 主要執行部分 ---
if __name__ == "__main__":
    data_directory = 'DATA'
    output_plots_directory = "OUTPUT_PLOTS" # 新增: 輸出圖片的資料夾名稱
    all_files_data = {}

    if not os.path.isdir(data_directory):
        print(f"錯誤：目錄 '{data_directory}' 不存在或不是一個目錄。")
        print(f"請在腳本所在位置建立 '{data_directory}' 目錄，並將 .xls 檔案放入其中。")
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

    successfully_processed_files = []
    for filepath in xls_file_paths:
        filename = os.path.basename(filepath) # 這是 .xls 檔案名
        print(f"\n正在處理檔案: {filename}...")

        time_data, acceleration_data = read_xls_columns_to_arrays(filepath)

        if time_data is not None and acceleration_data is not None:
            all_files_data[filename] = {
                'time': time_data,
                'acceleration': acceleration_data
            }
            successfully_processed_files.append(filename)
            print(f"  成功讀取檔案 '{filename}'。")
            print(f"    時間數據點數量: {len(time_data)}")
            print(f"    加速度數據點數量: {len(acceleration_data)}")
        else:
            print(f"  未能從檔案 '{filename}' 讀取所需的數據，已跳過。")

    print("\n" + "=" * 30)
    if not successfully_processed_files:
        print("沒有成功處理任何檔案，無法進行繪圖儲存。")
        exit()

    # --- 讓使用者選擇檔案並儲存圖片 ---
    print("已成功處理以下檔案：")
    for i, fname in enumerate(successfully_processed_files):
        print(f"  {i + 1}. {fname}")

    while True:
        try:
            choice_str = input(f"\n請輸入要產生圖片的檔案名稱 (或其編號，或輸入 'q' 離開): ").strip()
            if choice_str.lower() == 'q':
                print("使用者選擇離開。")
                break

            chosen_filename_xls = None # 確保這是 .xls 檔案名
            if choice_str.isdigit():
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(successfully_processed_files):
                    chosen_filename_xls = successfully_processed_files[choice_idx]
                else:
                    print("無效的編號選擇，請重新輸入。")
                    continue
            elif choice_str in successfully_processed_files:
                chosen_filename_xls = choice_str
            # 可以加上檢查，如果使用者輸入了 .png 結尾的檔名，提示他們輸入 .xls 檔名
            elif choice_str.endswith(".xls") and choice_str not in successfully_processed_files:
                print(f"錯誤：檔案 '{choice_str}' 不在已成功處理的列表中。")
                continue
            else:
                print(f"錯誤：輸入 '{choice_str}' 無效或檔案未在已處理列表中。請從上方列表選擇 .xls 檔案名稱或編號。")
                continue

            if chosen_filename_xls:
                print(f"\n正在為檔案 '{chosen_filename_xls}' 產生時間與加速度關係圖並儲存...")
                data_to_plot = all_files_data[chosen_filename_xls]
                save_time_acceleration_plot(data_to_plot['time'],
                                            data_to_plot['acceleration'],
                                            chosen_filename_xls, # 傳遞原始 .xls 檔案名
                                            output_plots_directory) # 傳遞輸出目錄

        except ValueError:
            print("輸入無效，請輸入檔案名稱、數字編號或 'q'。")
        except Exception as e:
            print(f"處理使用者輸入或產生圖片時發生錯誤: {e}")

    print("\n程式結束。")