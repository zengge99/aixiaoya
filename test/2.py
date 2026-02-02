import fsspec
import py7zr

def list_remote_7z(url):
    print(f"正在连接到远程服务器...")
    
    try:
        # 1. 使用 fsspec 打开远程 URL
        # fsspec 会自动处理 302 重定向
        # 'rb' 模式配合 fsspec 会创建一个支持随机访问 (seek) 的远程文件对象
        with fsspec.open(url, "rb") as remote_file:
            
            print("正在读取 7z 索引块 (仅下载必要部分)...")
            
            # 2. 将此文件对象直接传给 py7zr
            with py7zr.SevenZipFile(remote_file, mode='r') as archive:
                print(f"{'文件名':<60} | {'原始大小 (Bytes)':>15}")
                print("-" * 80)
                
                # 获取目录结构
                for file_info in archive.list():
                    print(f"{file_info.filename:<60} | {file_info.uncompressed:>15}")
                    
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 替换为你的 50GB 7z 下载链接
    url = "http://example.com/your_huge_file.7z"
    list_remote_7z(url)