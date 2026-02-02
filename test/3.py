import fsspec
import py7zr
import io
import sys
import os

# 保留原有偏移量文件包装类，无需修改
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

# 新增：纯内存解压单个文件的核心函数（最老py7zr兼容）
def extract_single_file_to_memory(archive, filename):
    """
    定向解压7z中单个文件到内存字节流，返回字节数据
    :param archive: 已打开的SevenZipFile对象
    :param filename: 压缩包内的文件相对路径
    :return: 字节数据/None
    """
    try:
        # 核心：创建内存字节流，作为解压目标
        bio = io.BytesIO()
        # 调用py7zr最底层的解压方法，定向解压单个文件到内存流
        # 适配所有py7zr版本的核心extract逻辑
        archive.extract(targets=[filename], path=bio)
        # 重置内存流指针到开头
        bio.seek(0)
        # 拼接内存流中的实际文件路径
        file_path = os.path.join(bio.name, filename)
        # 读取文件字节数据
        with open(file_path, 'rb') as f:
            data = f.read()
        # 清理内存临时文件
        os.remove(file_path)
        bio.close()
        return data
    except Exception as e:
        # 捕获单文件错误，不影响整体
        return None

def process_masked_7z_strm(url, offset, output_file="strm_out.txt", batch_size=1000):
    """
    修复0条写入 | 最老py7zr兼容 | 纯内存提取 | 不解压到磁盘 | 26万strm适配
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过偏移量 ({offset} 字节)...")
    total_strm = 0
    total_written = 0
    fail_count = 0

    try:
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            # 打开7z文件（全程保持打开，避免重复连接）
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                # 过滤strm文件，同时去重+过滤空路径
                strm_targets = [
                    f for f in all_files 
                    if f and f.lower().endswith('.strm') and os.path.basename(f)
                ]
                total_strm = len(strm_targets)

                if total_strm == 0:
                    print("未找到 .strm 文件，任务结束。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件 | 纯内存逐文件提取（每批{batch_size}条）...")
                # 清空输出文件
                open(output_file, "w", encoding="utf-8").close()

                with open(output_file, "a", encoding="utf-8") as f_out:
                    for idx, filename in enumerate(strm_targets):
                        # 纯内存提取单个文件
                        raw_data = extract_single_file_to_memory(archive, filename)
                        if raw_data is None or len(raw_data) == 0:
                            fail_count += 1
                            continue
                        
                        # 编码兼容处理（保持原有逻辑）
                        try:
                            content = raw_data.decode('utf-8').strip()
                        except Exception:
                            content = raw_data.decode('gbk', errors='ignore').strip()
                        
                        # 写入文件（原格式：文件名#内容）
                        f_out.write(f"{filename}#{content}\n")
                        total_written += 1

                        # 分批打印进度
                        if total_written % batch_size == 0:
                            print(f"进度：{total_written} / {total_strm} | 失败：{fail_count}")
                    
                    # 打印最终批次进度
                    if total_written % batch_size != 0:
                        print(f"进度：{total_written} / {total_strm} | 失败：{fail_count}")

                # 最终统计
                print(f"\n✅ 处理完成！")
                print(f"📊 统计：共找到{total_strm}个strm文件 | 成功写入{total_written}条 | 读取失败{fail_count}条")
                print(f"📁 结果文件：{os.path.abspath(output_file)} | 文件大小：{os.path.getsize(output_file)/1024/1024:.2f}MB")

    except Exception as e:
        print(f"\n❌ 全局错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # ==================== 仅修改这1行！====================
    TARGET_URL = "你的远程7z文件下载链接"  # 替换为实际链接
    # =====================================================
    REAL_OFFSET = 370745    # 偏移量，无需修改
    BATCH_SIZE = 1000       # 每批打印进度，内存大调2000/5000
    OUTPUT_FILE = "strm_out.txt"  # 输出文件名

    process_masked_7z_strm(
        url=TARGET_URL,
        offset=REAL_OFFSET,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE
    )