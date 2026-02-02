import fsspec
import py7zr
import io

# 7z 文件的标准特征码
SEVENZ_SIGNATURE = b'\x37\x7a\xbc\xaf\x27\x1c'

class OffsetFileWrapper(io.IOBase):
    """
    一个包装类，将文件操作透明地偏移到指定位置。
    让 py7zr 以为 offset 处就是文件开头。
    """
    def __init__(self, raw_file, offset):
        self.raw_file = raw_file
        self.offset = offset
        self.raw_file.seek(offset)
        # 获取逻辑上的文件总大小
        self.raw_file.seek(0, io.SEEK_END)
        self.total_size = self.raw_file.tell() - offset
        self.raw_file.seek(offset)

    def read(self, size=-1):
        return self.raw_file.read(size)

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            return self.raw_file.seek(self.offset + offset)
        elif whence == io.SEEK_CUR:
            return self.raw_file.seek(offset, io.SEEK_CUR)
        elif whence == io.SEEK_END:
            return self.raw_file.seek(offset, io.SEEK_END)
        return self.raw_file.tell() - self.offset

    def tell(self):
        return self.raw_file.tell() - self.offset

    def seekable(self): return True
    def readable(self): return True

def list_masked_7z(url):
    # 115 往往需要 Referer 或特定的 User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com", 
    }

    try:
        # 1. 使用 fsspec 打开远程文件
        print("正在连接服务器并搜索 7z 特征码...")
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            
            # 2. 扫描前 10MB 寻找 7z 签名 (通常伪装头不会超过这个范围)
            search_range = 10 * 1024 * 1024 
            initial_data = remote_file.read(search_range)
            
            real_start = initial_data.find(SEVENZ_SIGNATURE)
            
            if real_start == -1:
                print("错误：在扫描范围内未找到 7z 特征码。")
                return

            print(f"找到 7z 起始偏移量: {real_start} 字节")

            # 3. 使用包装器重定向文件流
            wrapped_file = OffsetFileWrapper(remote_file, real_start)
            
            # 4. 交给 py7zr 处理
            print("正在读取目录结构 (仅请求索引块)...")
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print(f"\n{'文件名':<60} | {'原始大小':>15}")
                print("-" * 80)
                for file_info in archive.list():
                    print(f"{file_info.filename:<60} | {file_info.uncompressed:>15}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 使用你提供的 URL
    url = "https://cdnfhnfile.115cdn.net/696b8ca0372d5bff49ca203b001741f315f0a042/115_20260202_153831.mp4?t=1772497203&u=334875423&s=524288000&d=vip-795368560-cq5tpbmhwf8epdmp4-1-0&c=2&f=1&k=00a45ce0b9b3f5619652b70dbcd9959c&us=5242880000&uc=10&v=1"
    list_masked_7z(url)