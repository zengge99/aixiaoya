import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import threading
import queue
from urllib.parse import urljoin, unquote
import time
import warnings

# --- 屏蔽烦人的警告 ---
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- 配置 ---
BASE_URL = "http://emby.xiaoya.pro/"
OUTPUT_FILE = "strm.txt"
THREAD_COUNT = 100  # 可以尝试增加到 20 提升速度

# --- 全局变量 ---
task_queue = queue.Queue()
visited_urls = set()
visited_lock = threading.Lock()
file_lock = threading.Lock()
stats = {"files": 0, "dirs": 0}
stats_lock = threading.Lock()

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        })
    return thread_local.session

def process_url(url):
    session = get_session()
    try:
        # 1. 如果是 .strm 文件，我们不需要用 BeautifulSoup 解析，直接获取内容
        if url.lower().endswith('.strm'):
            resp = session.get(url, timeout=10)
            # 使用 apparent_encoding 处理可能的乱码
            resp.encoding = resp.apparent_encoding 
            content = resp.text.strip().replace('\n', '').replace('\r', '')
            
            decoded_path = unquote(url)
            prefix = BASE_URL.rstrip('/')
            short_path = decoded_path.replace(prefix, "")

            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{short_path}#{content}\n")
            
            with stats_lock:
                stats["files"] += 1
            return

        # 2. 如果是目录，解析网页链接
        resp = session.get(url, timeout=10)
        # 强制设置编码为 utf-8，如果还是报错则忽略非法字符
        html_content = resp.content.decode('utf-8', errors='ignore')
        
        # 使用 lxml 解析器，速度极快
        soup = BeautifulSoup(html_content, 'lxml')
        
        with stats_lock:
            stats["dirs"] += 1

        for link in soup.find_all('a'):
            href = link.get('href')
            if not href or href.startswith('?') or href.startswith('/') or href == '../':
                continue
            
            full_url = urljoin(url, href)
            
            if full_url.startswith(BASE_URL):
                with visited_lock:
                    if full_url not in visited_urls:
                        visited_urls.add(full_url)
                        task_queue.put(full_url)

    except Exception:
        pass

def worker():
    while True:
        try:
            # 缩短 timeout，让程序在任务全部完成后更快退出
            current_url = task_queue.get(timeout=5)
            process_url(current_url)
            task_queue.task_done()
        except queue.Empty:
            break

def main():
    # 初始化文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print(f"🚀 启动快速爬虫 (LXML+Session)，线程数: {THREAD_COUNT}")
    start_time = time.time()

    visited_urls.add(BASE_URL)
    task_queue.put(BASE_URL)

    threads = []
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=worker)
        t.daemon = True # 设置为守护线程，方便主程序强制退出
        t.start()
        threads.append(t)

    # 监控进度
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with stats_lock:
                # 实时刷新进度
                print(f"进度: 文件夹 {stats['dirs']} | 找到 strm {stats['files']} | 剩余队列 {task_queue.qsize()}    ", end='\r')
        
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n🛑 用户强制停止")

    end_time = time.time()
    print(f"\n\n✅ 爬取完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"结果文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()