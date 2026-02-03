import requests
from bs4 import BeautifulSoup
import threading
import queue
from urllib.parse import urljoin, unquote
import time

# --- 配置 ---
BASE_URL = "http://emby.xiaoya.pro/"
OUTPUT_FILE = "strm.txt"
THREAD_COUNT = 10  # 10线程并发

# --- 全局变量 ---
task_queue = queue.Queue()
visited_urls = set()
visited_lock = threading.Lock()
file_lock = threading.Lock()
# 统计计数
stats = {"files": 0, "dirs": 0}
stats_lock = threading.Lock()

# 每个线程拥有自己的 session 提高效率（Keep-Alive）
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        # 设置通用的 User-Agent
        thread_local.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        })
    return thread_local.session

def process_url(url):
    session = get_session()
    try:
        # 统一使用 timeout 防止死锁
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return

        # 1. 处理 .strm 文件
        if url.lower().endswith('.strm'):
            content = resp.text.strip().replace('\n', '').replace('\r', '')
            # 路径处理：URL解码 -> 移除域名
            decoded_path = unquote(url)
            prefix = BASE_URL.rstrip('/')
            short_path = decoded_path.replace(prefix, "")

            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{short_path}#{content}\n")
            
            with stats_lock:
                stats["files"] += 1
            return

        # 2. 处理目录
        with stats_lock:
            stats["dirs"] += 1
            
        soup = BeautifulSoup(resp.content, 'html.parser')
        links = soup.find_all('a')
        
        for link in links:
            href = link.get('href')
            # 过滤逻辑
            if not href or href.startswith('?') or href.startswith('/') or href == '../':
                continue
            
            full_url = urljoin(url, href)
            
            # 确保只在站内爬取
            if full_url.startswith(BASE_URL):
                with visited_lock:
                    if full_url not in visited_urls:
                        visited_urls.add(full_url)
                        task_queue.put(full_url)

    except Exception as e:
        # 打印错误方便调试，实际运行时可注释掉
        # print(f"Error processing {url}: {e}")
        pass

def worker():
    """线程工作循环"""
    while True:
        try:
            # 只有当队列为空超过3秒时才退出，确保动态生成的任务有时间入队
            current_url = task_queue.get(timeout=3)
            process_url(current_url)
            task_queue.task_done()
        except queue.Empty:
            # 队列空了，代表任务可能已经完成
            break

def main():
    # 初始化文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print(f"🚀 启动爬虫，线程数: {THREAD_COUNT}")
    start_time = time.time()

    # 放入初始任务
    visited_urls.add(BASE_URL)
    task_queue.put(BASE_URL)

    # 创建并启动线程
    threads = []
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # 监控进度
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        with stats_lock:
            print(f"进度监控: 已扫描目录 {stats['dirs']}, 已提取 strm 文件 {stats['files']} ...", end='\r')

    # 等待所有线程结束
    for t in threads:
        t.join()

    end_time = time.time()
    print(f"\n\n✅ 爬取完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"总计提取: {stats['files']} 个 strm 文件")
    print(f"结果文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户手动停止。")