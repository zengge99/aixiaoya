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

# 核心：内存高效的提取工厂
class StrmExtractorFactory:
    def __init__(self, output_handle, total_count):
        self.output_handle = output_handle
        self.total_count = total_count
        self.current_count = 0

    def get_target(self, member):
        """每当 py7zr 准备提取一个文件时，会调用此方法"""
        # 返回一个 BytesIO 作为临时中转，用于接收当前这一个文件的内容
        return self.StrmFileBuffer(member.filename, self)

    class StrmFileBuffer(io.BytesIO):
        """内部类：负责接收单个文件数据，并在提取完成后立即写入最终文件"""
        def __init__(self, filename, outer_factory):
            super().__init__()
            self.filename = filename
            self.outer = outer_factory

        def close(self):
            """当 py7zr 完成当前文件的解压写入后，会调用 close()"""
            raw_content = self.getvalue()
            if raw_content:
                # 尝试解码
                try:
                    content = raw_content.decode('utf-8').strip()
                except:
                    content = raw_content.decode('gbk', errors='ignore').strip()
                
                # 立即写入最终文件，释放内存
                self.outer.output_handle.write(f"{self.filename}#{content}\n")
            
            self.outer.current_count += 1
            if self.outer.current_count % 5000 == 0:
                print(f"已处理 {self.outer.current_count} / {self.outer.total_count} 个文件...")
            
            super().close()

def process_masked_7z_strm(url, offset, output_file="strm_out.txt"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过偏移量 ({offset} 字节)...")
    
    try:
        # 1. 打开远程文件
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            # 2. 打开 7z 索引
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

                # 3. 创建输出文件并开始流式解压
                with open(output_file, "w", encoding="utf-8") as f_out:
                    # 初始化工厂
                    factory = StrmExtractorFactory(f_out, total_strm)
                    # 执行提取：py7zr 会逐个解压 targets 中的文件，并通过 factory 处理
                    archive.extract(targets=strm_targets, factory=factory)
                
                print(f"处理成功！所有数据已保存至 {output_file}。")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 配置区
    TARGET_URL = "你的下载链接"
    REAL_OFFSET = 370745
    OUTPUT_NAME = "strm_out.txt"
    
    process_masked_7z_strm(TARGET_URL, REAL_OFFSET, OUTPUT_NAME)