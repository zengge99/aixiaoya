import fsspec
import py7zr
import io
# 导入高层 API
from py7zr import extract_7z_archive

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
        # 1. 使用 fsspec 打开远程文件
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            
            # 2. 包装文件流
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            
            # 3. 先通过 SevenZipFile 获取文件列表
            print("正在读取索引以匹配 .strm 文件...")
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
            
            total_strm = len(strm_targets)
            if total_strm == 0:
                print("未找到 .strm 文件。")
                return
            
            print(f"找到 {total_strm} 个目标文件。")
            
            # --- 关键修改：回到起点，使用 extract_7z_archive 处理内存流 ---
            wrapped_file.seek(0) 
            print("正在通过内存提取数据（不落地磁盘）...")
            
            # extract_7z_archive 直接接受 file-like object (wrapped_file)
            # targets 参数指定只解压我们需要的文件
            # 返回值格式：{文件名: BytesIO内容}
            extracted_data = extract_7z_archive(wrapped_file, targets=strm_targets)
            
            # 4. 写入结果文件
            print(f"提取完成，正在格式化并写入 {output_file}...")
            count = 0
            with open(output_file, "w", encoding="utf-8") as f_out:
                for name in strm_targets:
                    if name in extracted_data:
                        # 从 BytesIO 中读取字节并解码
                        raw_bytes = extracted_data[name].getbuffer()
                        try:
                            content = raw_bytes.tobytes().decode('utf-8').strip()
                        except:
                            content = raw_bytes.tobytes().decode('gbk', errors='ignore').strip()
                        
                        f_out.write(f"{name}#{content}\n")
                        count += 1
            
            print(f"处理成功！已完成 {count} 个文件的内容提取。")

    except Exception as e:
        print(f"\n解析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    TARGET_URL = "你的下载链接"
    REAL_OFFSET = 370745
    
    process_masked_7z_strm(TARGET_URL, REAL_OFFSET)