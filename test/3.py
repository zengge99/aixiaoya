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
    适配最古老py7zr | 仅用getnames | 纯内存提取strm | 不解压磁盘 | 分批写入
    无任何高级方法依赖，是py7zr最低兼容版本
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
            # 核心：仅用open和getnames，这是所有py7zr版本必带的
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                total_strm = len(strm_targets)

                if total_strm == 0:
                    print("未找到 .strm 文件，任务结束。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件 | 纯内存逐文件提取（每批{batch_size}条）...")
                # 清空输出文件
                open(output_file, "w", encoding="utf-8").close()

                with open(output_file, "a", encoding="utf-8") as f_out:
                    # 核心逻辑：逐文件处理，利用py7zr底层解压流，纯内存读取
                    for idx, filename in enumerate(strm_targets):
                        try:
                            # 最底层方式：直接通过7z归档的解压流读取单个文件，纯内存
                            # 适配所有py7zr版本的核心解压逻辑，无封装方法依赖
                            for name, content in archive._SevenZipFile__archive.iterfiles([filename]):
                                # content是字节流，直接读取，不解压到磁盘
                                raw_content = content.read()
                                # 编码兼容，保持原有逻辑
                                try:
                                    content_str = raw_content.decode('utf-8').strip()
                                except:
                                    content_str = raw_content.decode('gbk', errors='ignore').strip()
                                # 按原格式写入
                                f_out.write(f"{name}#{content_str}\n")
                                total_written += 1
                                # 关闭内存流，释放资源
                                content.close()
                            # 打印进度
                            if total_written % batch_size == 0 and total_written > 0:
                                print(f"已写入 {total_written} / {total_strm} 条记录...")
                        except Exception as e:
                            # 跳过单个文件读取错误，不影响整体
                            continue

                # 最终统计
                print(f"\n✅ 处理完成！")
                print(f"📊 统计：共找到{total_strm}个strm文件 | 成功写入{total_written}条有效数据")
                print(f"📁 结果文件：{os.path.abspath(output_file)}")
                if total_written < total_strm:
                    print(f"⚠️  提示：有{total_strm - total_written}个文件读取失败，已自动跳过")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # ==================== 仅修改这1行！====================
    TARGET_URL = "你的远程7z文件下载链接"
    # =====================================================
    REAL_OFFSET = 370745    # 偏移量，无需修改
    BATCH_SIZE = 1000       # 每批写入条数，内存小调500，内存大调2000
    OUTPUT_FILE = "strm_out.txt"  # 输出文件名

    process_masked_7z_strm(
        url=TARGET_URL,
        offset=REAL_OFFSET,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE
    )