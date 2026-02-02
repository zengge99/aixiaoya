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
        # 1. 使用 fsspec 打开远程文件
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            
            # 2. 包装文件流
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            # 3. 解析 7z
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取 7z 索引，正在筛选 .strm 文件...")
                
                # 获取所有文件名并筛选出 .strm
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                
                if not strm_targets:
                    print("未在压缩包中找到 .strm 文件。")
                    return

                print(f"找到 {len(strm_targets)} 个 .strm 文件，正在提取内容...")
                
                # 4. 仅读取选定的 strm 文件
                # read() 返回一个字典 {文件名: BytesIO对象}
                extracted_data = archive.read(targets=strm_targets)
                
                # 5. 写入本地文件
                count = 0
                with open(output_file, "w", encoding="utf-8") as f_out:
                    for name in strm_targets:
                        if name in extracted_data:
                            # 获取二进制内容并转换为字符串，去掉换行符
                            raw_content = extracted_data[name].read()
                            try:
                                content = raw_content.decode('utf-8').strip()
                            except UnicodeDecodeError:
                                # 如果 utf-8 解码失败，尝试 gbk 或 忽略错误
                                content = raw_content.decode('gbk', errors='ignore').strip()
                            
                            # 写入格式：全路径名#文件内容
                            f_out.write(f"{name}#{content}\n")
                            count += 1
                
                print(f"处理完成！已将 {count} 条记录写入到 {output_file}")

    except Exception as e:
        print(f"\n解析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 配置参数
    TARGET_URL = "你的下载链接"
    REAL_OFFSET = 370745  # 你的偏移量
    
    process_masked_7z_strm(TARGET_URL, REAL_OFFSET)