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

def process_masked_7z_strm(url, offset, output_file="strm_out.txt", batch_size=1000):
    """
    超老版py7zr兼容 | 纯内存提取远程7z中strm | 不解压磁盘 | 分批写入
    :param batch_size: 每批写入条数（根据服务器内存调整，默认1000）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://115.com",
    }

    print(f"正在连接服务器，跳过偏移量 ({offset} 字节)...")
    total_strm = 0
    total_written = 0

    try:
        with fsspec.open(url, "rb", headers=headers) as remote_file:
            wrapped_file = OffsetFileWrapper(remote_file, offset)
            # 核心：打开7z后，直接通过py7zr的底层文件迭代器读取，无高级方法依赖
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                total_strm = len(strm_targets)

                if total_strm == 0:
                    print("未找到 .strm 文件，任务结束。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件 | 纯内存提取+分批写入（每批{batch_size}条）...")
                # 清空输出文件，避免追加旧内容
                open(output_file, "w", encoding="utf-8").close()

                with open(output_file, "a", encoding="utf-8") as f_out:
                    # 遍历7z底层文件流，仅处理strm文件，纯内存读取
                    for entry in archive.archive.getmembers():
                        # 过滤非strm文件和不在目标列表的文件
                        if not entry.filename.lower().endswith('.strm') or entry.filename not in strm_targets:
                            continue
                        # 纯内存读取当前文件内容（核心底层操作，无磁盘写入）
                        raw_content = archive.readfile(entry)
                        # 编码兼容：保持你原有逻辑（utf-8优先，失败则gbk忽略错误）
                        try:
                            content = raw_content.decode('utf-8').strip()
                        except Exception:
                            content = raw_content.decode('gbk', errors='ignore').strip()
                        # 按原有格式写入：文件名#内容
                        f_out.write(f"{entry.filename}#{content}\n")
                        total_written += 1

                        # 打印进度（每batch_size条输出一次）
                        if total_written % batch_size == 0:
                            print(f"已写入 {total_written} / {total_strm} 条记录...")

                # 最终统计结果
                print(f"\n✅ 处理成功！")
                print(f"📊 统计：共找到{total_strm}个strm文件 | 成功写入{total_written}条有效数据")
                print(f"📁 结果文件：{os.path.abspath(output_file)}")
                if total_written < total_strm:
                    print(f"⚠️  提示：有{total_strm - total_written}个文件未读取，已自动跳过")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 仅需修改这1个配置！替换为你的实际远程7z下载链接
    TARGET_URL = "你的远程7z文件下载链接"
    # 以下参数保持不变
    REAL_OFFSET = 370745    # 偏移量（无需修改）
    BATCH_SIZE = 1000       # 每批写入条数，内存小则调小（如500）
    OUTPUT_FILE = "strm_out.txt"  # 输出文件名

    process_masked_7z_strm(
        url=TARGET_URL,
        offset=REAL_OFFSET,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE
    )