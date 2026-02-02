import fsspec
import py7zr
import io
import sys

# 处理 115 等特殊偏移量的文件包装器
class OffsetFileWrapper(io.IOBase):
    def __init__(self, raw_file, offset):
        self.raw_file = raw_file
        self.offset = offset
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

# 修正后的工厂类：解决 'create' 属性报错问题
class StrmExtractorFactory:
    def __init__(self, output_handle, total_count):
        self.output_handle = output_handle
        self.total_count = total_count
        self.current_count = 0

    def create(self, filename):
        """
        py7zr 最新 API 要求使用 create 方法。
        filename: 压缩包内的文件路径
        """
        # 返回一个内存缓冲对象处理单个文件内容
        return self.StrmFileBuffer(filename, self)

    class StrmFileBuffer(io.BytesIO):
        def __init__(self, filename, outer_factory):
            super().__init__()
            self.filename = filename
            self.outer = outer_factory

        def close(self):
            """当 py7zr 解压完当前文件并调用 close() 时，我们将内容写入最终文件"""
            raw_content = self.getvalue()
            if raw_content:
                try:
                    content = raw_content.decode('utf-8').strip()
                except:
                    # 备选编码处理
                    content = raw_content.decode('gbk', errors='ignore').strip()
                
                # 实时写入磁盘，避免 26 万个内容堆积在内存里
                self.outer.output_handle.write(f"{self.filename}#{content}\n")
            
            self.outer.current_count += 1
            if self.outer.current_count % 5000 == 0:
                print(f"已处理 {self.outer.current_count} / {self.outer.total_count}...")
            
            super().close()

def process_masked_7z_strm(url, offset, output_file="strm_out.txt"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过偏移量 ({offset} 字节)...")
    
    try:
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            # 使用 context manager 确保资源释放
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                
                total_strm = len(strm_targets)
                if total_strm == 0:
                    print("未找到 .strm 文件。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件。")
                print("正在流式提取（内存友好模式）...")

                # 核心：使用 'create' 工厂模式进行流式解压
                with open(output_file, "w", encoding="utf-8") as f_out:
                    factory = StrmExtractorFactory(f_out, total_strm)
                    # 这里的 factory 参数会调用我们写的 create 方法
                    archive.extract(targets=strm_targets, factory=factory)
                
                print(f"\n处理成功！所有数据已保存至 {output_file}。")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 配置
    TARGET_URL = "你的下载链接"
    REAL_OFFSET = 370745
    
    process_masked_7z_strm(TARGET_URL, REAL_OFFSET)