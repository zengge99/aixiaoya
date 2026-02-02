import fsspec
import py7zr
import io
import sys

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
    纯内存处理远程7z文件，提取strm内容到本地文件（不解压到磁盘）
    :param batch_size: 每批写入文件数，根据内存调整（默认1000）
    :param url: 远程7z文件链接
    :param offset: 跳过的字节偏移量
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
            with py7zr.SevenZipFile(wrapped_file, mode='r') as archive:
                print("成功读取索引，正在检索文件名...")
                all_files = archive.getnames()
                # 筛选所有strm文件（压缩包内相对路径）
                strm_targets = [f for f in all_files if f.lower().endswith('.strm')]
                total_strm = len(strm_targets)

                if total_strm == 0:
                    print("未找到 .strm 文件，任务结束。")
                    return

                print(f"找到 {total_strm} 个 .strm 文件，纯内存提取中（不解压磁盘）...")
                # 核心：纯内存读取所有文件对象（readall是py7zr全版本原生方法）
                # 返沪字典：{压缩包内文件路径: 内存文件对象}，全程无磁盘写入
                all_in_memory_files = archive.readall()

                # 清空输出文件（避免追加旧内容）
                open(output_file, "w", encoding="utf-8").close()
                print(f"开始分批写入结果到 {output_file}（每批{batch_size}条）...")

                # 分批处理+写入，避免一次性加载所有内容占满内存
                with open(output_file, "a", encoding="utf-8") as f_out:
                    for i in range(0, total_strm, batch_size):
                        # 切分当前批次的strm文件
                        batch_files = strm_targets[i:i+batch_size]
                        batch_num = i // batch_size + 1
                        total_batch = (total_strm + batch_size - 1) // batch_size

                        # 处理当前批次，纯内存读取内容
                        for name in batch_files:
                            if name in all_in_memory_files:
                                # 从内存文件对象读取字节数据，不解压到磁盘
                                raw_content = all_in_memory_files[name].read()
                                # 编码兼容：先utf-8，失败则gbk忽略错误（保持原有逻辑）
                                try:
                                    content = raw_content.decode('utf-8').strip()
                                except Exception:
                                    content = raw_content.decode('gbk', errors='ignore').strip()
                                # 按原有格式写入：文件名#内容
                                f_out.write(f"{name}#{content}\n")
                                total_written += 1

                        # 打印批次进度
                        print(f"批次 {batch_num}/{total_batch} 完成 | 已累计写入 {total_written} / {total_strm} 条记录...")

                # 最终结果统计
                print(f"\n✅ 处理成功！")
                print(f"📊 统计：共找到{total_strm}个strm文件 | 成功写入{total_written}条有效数据")
                print(f"📁 结果文件：{os.path.abspath(output_file)}")
                if total_written < total_strm:
                    print(f"⚠️  提示：有{total_strm - total_written}个文件未读取到，已自动跳过")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # ==================== 仅需修改这里的配置 ====================
    TARGET_URL = "你的远程7z文件下载链接"  # 替换为你的实际7z链接
    REAL_OFFSET = 370745  # 偏移量，保持原有数值不变
    BATCH_SIZE = 1000     # 每批处理数，内存小则调小（如500），内存大则调大（如2000/5000）
    OUTPUT_FILE = "strm_out.txt"  # 输出结果文件名
    # ============================================================

    # 执行主函数
    process_masked_7z_strm(
        url=TARGET_URL,
        offset=REAL_OFFSET,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE
    )