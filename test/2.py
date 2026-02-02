import py7zr
from httpio import httpreader
import requests

def list_remote_7z(url):
    # 1. 处理 302 重定向并获取最终 URL（可选，httpio 内部通常也能处理）
    # 但显式处理可以确保我们拿到最终地址
    response = requests.head(url, allow_redirects=True)
    final_url = response.url
    print(f"最终访问地址: {final_url}")

    try:
        # 2. 使用 httpio 创建一个远程文件句柄
        # 它支持随机读取 (seekable)，这对于读取 7z 的元数据至关重要
        with httpreader(final_url) as remote_file:
            # 3. 将远程文件句柄传给 py7zr
            with py7zr.SevenZipFile(remote_file, mode='r') as archive:
                print("成功读取元数据，目录结构如下：\n")
                
                # 获取所有文件信息
                for file_info in archive.list():
                    # 打印文件名和原始大小
                    # file_info 包含 filename, uncompressed, compressed 等属性
                    print(f"{file_info.filename:<50} {file_info.uncompressed:>12} bytes")

    except Exception as e:
        print(f"解析失败: {e}")

if __name__ == "__main__":
    # 替换为你的 50GB 7z 文件链接
    target_url = "http://example.com/large_archive.7z"
    list_remote_7z(target_url)