import fsspec
import py7zr
import io
import sys

# 尝试解决版本兼容性问题
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

def process_masked_7z_strm(url, offset, output_file="strm_out.txt"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过偏移量 ({offset} 字节)...")
    
    try:
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                
                total_strm = len(strm_targets)
                if total_strm == 0:
                    print("未找到 .strm 文件。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件。")
                print("正在提取内容（文件较多，请耐心等待，这可能需要较长时间）...")

                # 兼容性处理：尝试不同的读取方法
                extract_func = None
                if hasattr(archive, 'read'):
                    extract_func = archive.read
                elif hasattr(archive, 'get_data'):
                    extract_func = archive.get_data
                
                if extract_func is None:
                    print("错误: 无法在 py7zr 中找到读取方法，请尝试运行 'pip install --upgrade py7zr'")
                    return

                # 提取数据
                # 注意：26万个文件在这里可能会消耗大量内存和时间
                extracted_data = extract_func(targets=strm_targets)
                
                print(f"提取完成，正在写入文件 {output_file}...")
                
                count = 0
                with open(output_file, "w", encoding="utf-8") as f_out:
                    for name in strm_targets:
                        if name in extracted_data:
                            raw_content = extracted_data[name].read()
                            try:
                                content = raw_content.decode('utf-8').strip()
                            except:
                                content = raw_content.decode('gbk', errors='ignore').strip()
                            
                            f_out.write(f"{name}#{content}\n")
                            count += 1
                            if count % 10000 == 0:
                                print(f"已写入 {count} / {total_strm} 条记录...")
                
                print(f"处理成功！总计写入 {count} 条数据。")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 请确保已替换为实际链接
    TARGET_URL = "你的下载链接"
    REAL_OFFSET = 370745
    
    process_masked_7z_strm(TARGET_URL, REAL_OFFSET)