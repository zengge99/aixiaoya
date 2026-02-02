import requests
from remotezip import RemoteZip
import io

def merge_remote_zip_to_txt(url, output_path, encoding='utf-8'):
    # 1. 处理重定向，获取最终直链
    print(f"正在检查 URL 并处理可能的跳转...")
    try:
        # 使用 allow_redirects=True 获取最终地址
        # head 请求只拿响应头，不下载内容，速度快
        response = requests.head(url, allow_redirects=True, timeout=10)
        final_url = response.url
        if final_url != url:
            print(f"检测到跳转，最终地址: {final_url}")
        else:
            print("未检测到跳转，使用原始地址。")
            
        # 检查服务器是否支持 Range（断点续传/部分读取）
        if 'Accept-Ranges' not in response.headers and 'accept-ranges' not in response.headers:
            print("提示：服务器可能不支持 Range 请求，性能可能会受影响或报错。")

    except Exception as e:
        print(f"连接失败: {e}")
        return

    # 2. 流式读取并写入大文件
    count = 0
    try:
        # 开启 RemoteZip
        with RemoteZip(final_url) as rzf, \
             open(output_path, 'w', encoding=encoding, buffering=1024*1024) as out_f:
            
            print("正在扫描文件列表...")
            all_files = rzf.infolist()
            print(f"开始处理，共计 {len(all_files)} 个条目...")

            for member in all_files:
                # 过滤掉文件夹
                if member.is_dir():
                    continue
                
                try:
                    # 通过网络流读取该小文件的内容
                    with rzf.open(member) as f:
                        # 读取并解码，去掉多余空格/换行
                        content = f.read().decode(encoding).strip()
                        
                        # 格式：文件名#内容\n
                        out_f.write(f"{member.filename}#{content}\n")
                        
                        count += 1
                        if count % 5000 == 0:
                            print(f"进度: 已处理 {count} / {len(all_files)}")
                            
                except UnicodeDecodeError:
                    # 如果某些文件不是 UTF-8，可忽略或尝试 GBK
                    continue
                except Exception as e:
                    print(f"\n跳过文件 {member.filename}: {e}")

    except Exception as e:
        print(f"读取压缩包出错: {e}")

    print(f"\n处理完成！共存入 {count} 行数据至 {output_path}")

# --- 使用示例 ---
url = "http://your-server.com/data_redirect_link.zip" 
output_file = "all_in_one.txt"
merge_remote_zip_to_txt(url, output_file)