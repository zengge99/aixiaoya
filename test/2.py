import fsspec
import py7zr
import io

class OffsetFileWrapper(io.IOBase):
    """
    将原始文件流偏移指定字节，使其看起来像一个从 0 开始的新文件。
    """
    def __init__(self, raw_file, offset):
        self.raw_file = raw_file
        self.offset = offset
        # 初始化时直接跳到偏移位置
        self.raw_file.seek(offset)
        
    def read(self, size=-1):
        return self.raw_file.read(size)

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            # 绝对路径跳转，加上偏移量
            return self.raw_file.seek(self.offset + offset)
        elif whence == io.SEEK_CUR:
            # 相对当前位置跳转
            return self.raw_file.seek(offset, io.SEEK_CUR)
        elif whence == io.SEEK_END:
            # 相对文件末尾跳转（7z 索引通常在末尾）
            return self.raw_file.seek(offset, io.SEEK_END)
        return self.raw_file.tell() - self.offset

    def tell(self):
        # 返回逻辑上的位置（扣除偏移量）
        return self.raw_file.tell() - self.offset

    def seekable(self): return True
    def readable(self): return True

def list_masked_7z(url, offset):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过伪装头 ({offset} 字节)...")
    
    try:
        # 1. 使用 fsspec 打开远程文件，设置较小的 block_size 避免无谓下载
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            
            # 2. 包装文件流，使其从偏移量处作为逻辑起点
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            # 3. 交给 py7zr 解析
            print("正在请求 7z 索引块...")
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print(f"\n{'文件名':<60} | {'原始大小 (Bytes)':>15}")
                print("-" * 80)
                
                # 获取列表
                for file_info in archive.list():
                    print(f"{file_info.filename:<60} | {file_info.uncompressed:>15}")
                
                print("-" * 80)
                print("目录读取完成。")

    except Exception as e:
        print(f"\n解析失败: {e}")
        print("请检查：1. 链接是否过期；2. 服务器是否支持 Range 请求；3. 偏移量是否绝对准确。")

if __name__ == "__main__":
    # 你的 50GB 下载链接
    target_url = "你的下载链接"
    
    # 你指定的偏移量
    REAL_OFFSET = 370745
    
    list_masked_7z(target_url, REAL_OFFSET)