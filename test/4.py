import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urljoin, unquote
import time

# 配置
BASE_URL = "http://emby.xiaoya.pro/"
OUTPUT_FILE = "strm.txt"
MAX_WORKERS = 10  # 10线程并发

# 全局变量
visited_dirs = set()
results = []
results_lock = threading.Lock()
dir_lock = threading.Lock()

def fetch_url(url):
    """获取URL内容，带重试机制"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def process_page(url):
    """处理目录页或strm文件"""
    print(f"Scanning: {unquote(url)}")
    resp = fetch_url(url)
    if not resp:
        return

    # 如果是 .strm 文件，直接读取内容
    if url.lower().endswith('.strm'):
        content = resp.text.strip().replace('\n', ' ').replace('\r', '')
        with results_lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                line = f"{url}#{content}\n"
                f.write(line)
        return

    # 如果是目录，解析其中的链接
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = soup.find_all('a')
    
    sub_tasks = []
    
    for link in links:
        href = link.get('href')
        if not href or href.startswith('?') or href.startswith('/') or href == '../':
            continue
        
        full_url = urljoin(url, href)
        
        # 排除外部链接
        if not full_url.startswith(BASE_URL):
            continue

        if href.endswith('/'):
            # 目录：添加到待处理列表
            with dir_lock:
                if full_url not in visited_dirs:
                    visited_dirs.add(full_url)
                    sub_tasks.append(full_url)
        elif href.lower().endswith('.strm'):
            # strm文件：添加到任务
            sub_tasks.append(full_url)
            
    return sub_tasks

def main():
    # 初始化输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    # 使用线程池管理任务
    # 使用 set 来管理待处理任务，防止重复
    todo_urls = {BASE_URL}
    visited_dirs.add(BASE_URL)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        active_futures = set()
        
        # 提交初始任务
        active_futures.add(executor.submit(process_page, BASE_URL))

        while active_futures:
            # 等待任一任务完成
            done, active_futures = wait_for_any(active_futures)
            
            for future in done:
                try:
                    new_urls = future.result()
                    if new_urls:
                        for next_url in new_urls:
                            active_futures.add(executor.submit(process_page, next_url))
                except Exception as e:
                    print(f"Worker generated an exception: {e}")

def wait_for_any(futures):
    """辅助函数，等待集合中至少一个future完成"""
    from concurrent.futures import wait, FIRST_COMPLETED
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return done, not_done

if __name__ == "__main__":
    start_time = time.time()
    print("Starting crawl...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    print(f"\nFinished! Results saved to {OUTPUT_FILE}")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")