import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import threading
import queue
from urllib.parse import urljoin, unquote
import time
import warnings

# 屏蔽解析警告
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- 配置 ---
BASE_URL = "http://emby.xiaoya.pro/"
OUTPUT_FILE = "strm.txt"
THREAD_COUNT = 100 

# --- 全局变量 ---
task_queue = queue.Queue()
visited_urls = set()
visited_lock = threading.Lock()
file_lock = threading.Lock()
stats = {"files": 0, "dirs": 0, "skipped": 0}
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
    
    # 1. 如果是 .strm 文件：抓取内容并存盘
    if url.lower().endswith('.strm'):
        try:
            resp = session.get(url, timeout=10)
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
        except:
            pass
        return

    # 2. 如果是目录 (以 / 结尾) 或 根路径：解析下一级
    if url.endswith('/') or url == BASE_URL:
        try:
            resp = session.get(url, timeout=10)
            html_content = resp.content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'lxml')
            
            with stats_lock:
                stats["dirs"] += 1

            for link in soup.find_all('a'):
                href = link.get('href')
                # 过滤掉无效链接
                if not href or href.startswith('?') or href.startswith('/') or href == '../':
                    continue
                
                full_url = urljoin(url, href)
                
                # --- 核心改进：只放行目录和strm ---
                is_dir = href.endswith('/')
                is_strm = href.lower().endswith('.strm')
                
                if (is_dir or is_strm) and full_url.startswith(BASE_URL):
                    with visited_lock:
                        if full_url not in visited_urls:
                            visited_urls.add(full_url)
                            task_queue.put(full_url)
                else:
                    # 其他文件（mp4, mkv, jpg等）直接无视
                    pass
        except:
            pass
        return

def worker():
    while True:
        try:
            current_url = task_queue.get(timeout=5)
            process_url(current_url)
            task_queue.task_done()
        except queue.Empty:
            break

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print(f"🚀 启动定向爬虫，线程数: {THREAD_COUNT}")
    start_time = time.time()

    visited_urls.add(BASE_URL)
    task_queue.put(BASE_URL)

    threads = []
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with stats_lock:
                print(f"进度: 目录 {stats['dirs']} | strm {stats['files']} | 队列 {task_queue.qsize()}   ", end='\r')
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n停止")

    print(f"\n\n✅ 完成！耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()