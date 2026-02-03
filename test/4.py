import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
from urllib.parse import urljoin, unquote
import time

# --- 配置参数 ---
BASE_URL = "http://emby.xiaoya.pro/"
OUTPUT_FILE = "strm.txt"
MAX_WORKERS = 10  # 10线程并发

# --- 全局变量 ---
visited_urls = set()
visited_lock = threading.Lock()
file_lock = threading.Lock()

def fetch_content(url):
    """请求网页或文件内容"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response
    except Exception as e:
        # print(f"请求失败: {unquote(url)}") # 调试时可开启
        pass
    return None

def process_url(url):
    """处理单个URL"""
    # 1. 如果是 .strm 文件
    if url.lower().endswith('.strm'):
        resp = fetch_content(url)
        if resp:
            # 读取strm内容，去掉换行符
            content = resp.text.strip().replace('\n', '').replace('\r', '')
            
            # --- 路径处理逻辑 ---
            # 先解码 URL (转为中文等)
            decoded_full_url = unquote(url)
            # 删除开头的 http://emby.xiaoya.pro (注意去除末尾斜杠的影响)
            prefix = BASE_URL.rstrip('/')
            short_path = decoded_full_url.replace(prefix, "")
            
            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{short_path}#{content}\n")
            print(f"已抓取: {short_path}")
        return []

    # 2. 如果是目录 (解析 HTML 提取更多链接)
    resp = fetch_content(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    found_urls = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        # 过滤无效链接
        if not href or href.startswith('?') or href.startswith('/') or href == '../':
            continue
        
        full_url = urljoin(url, href)
        
        # 确保只在站内爬取
        if not full_url.startswith(BASE_URL):
            continue

        with visited_lock:
            if full_url not in visited_urls:
                visited_urls.add(full_url)
                found_urls.append(full_url)
                
    return found_urls

def main():
    # 初始化/清空文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print(f"开始爬取任务，并发线程数: {MAX_WORKERS}")
    visited_urls.add(BASE_URL)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将根目录加入队列
        futures = {executor.submit(process_url, BASE_URL)}
        
        while futures:
            # 等待至少一个线程完成
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            
            for future in done:
                try:
                    new_urls = future.result()
                    if new_urls:
                        for next_url in new_urls:
                            futures.add(executor.submit(process_url, next_url))
                except Exception as e:
                    pass # 忽略单个线程的报错

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户强制退出。")
    
    print("\n" + "="*30)
    print(f"爬取结束！")
    print(f"结果文件: {OUTPUT_FILE}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    print("="*30)