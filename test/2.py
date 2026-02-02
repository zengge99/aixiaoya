from remotezip import RemoteZip
import requests

def list_remote_zip(url):
    try:
        # remotezip 底层使用 requests.get()，默认 allow_redirects=True
        # 因此 302 跳转会被自动处理。
        # 对于 50GB 的大文件，remotezip 只会请求几百 KB 的元数据。
        with RemoteZip(url) as rz:
            print(f"成功连接到文件。正在解析目录结构...\n")
            
            # 打印类似于 'unzip -l' 的列表结构
            rz.printdir()
            
            # 如果你需要以列表形式处理，可以使用 infolist()
            # for file_info in rz.infolist():
            #     print(f"文件名: {file_info.filename}, 大小: {file_info.file_size} bytes")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 替换为你的 50GB HTTP 下载链接
    zip_url = "http://example.com/very_large_file.zip"
    list_remote_zip(zip_url)