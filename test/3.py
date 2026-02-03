import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import threading
import queue
from urllib.parse import urljoin, unquote, urlparse
import time
import warnings
import re
import xml.etree.ElementTree as ET

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

# 缓存目录对应的 tvshow.nfo 数据，避免重复请求
# 结构: { "dir_url": { "title": "xxx", "year": "xxx", "tmdb": "xxx" } }
tvshow_cache = {}
cache_lock = threading.Lock()

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        })
    return thread_local.session

def get_xml_tag_text(root, tag):
    """从XML中获取标签内容"""
    node = root.find(tag)
    if node is not None:
        return node.text
    if tag == "tmdbid": # 兼容某些nfo格式
        for uid in root.findall('uniqueid'):
            if uid.get('type') == 'tmdb':
                return uid.text
    return ""

def parse_nfo_data(url):
    """请求并解析NFO文件"""
    session = get_session()
    try:
        resp = session.get(url, timeout=5)
        if resp.status_code == 200:
            # 清理可能存在的编码问题
            content = resp.content.strip()
            root = ET.fromstring(content)
            data = {
                "title": get_xml_tag_text(root, "title"),
                "year": get_xml_tag_text(root, "year") or (get_xml_tag_text(root, "premiered")[:4] if get_xml_tag_text(root, "premiered") else ""),
                "tmdbid": get_xml_tag_text(root, "tmdbid") or get_xml_tag_text(root, "tmdb"),
                "season": get_xml_tag_text(root, "season"),
                "episode": get_xml_tag_text(root, "episode")
            }
            return data
    except:
        pass
    return None

def extract_resolution(path):
    """从路径中提取分辨率"""
    match = re.search(r'(2160p|1080p|720p|4k|540p)', path, re.I)
    return match.group(0).lower() if match else ""

def get_new_strm_path(url, strm_content):
    """根据NFO逻辑重构路径"""
    session = get_session()
    decoded_url = unquote(url)
    path_parts = decoded_url.replace(BASE_URL, "").strip('/').split('/')
    
    filename = path_parts[-1]
    parent_dir_url = url.rsplit('/', 1)[0] + '/'
    
    # 提取分辨率
    res = extract_resolution(decoded_url)
    res_suffix = f".{res}" if res else ""

    # 1. 尝试获取剧集信息 (tvshow.nfo)
    tvshow_info = None
    with cache_lock:
        if parent_dir_url in tvshow_cache:
            tvshow_info = tvshow_cache[parent_dir_url]
    
    if tvshow_info is None:
        # 尝试下载 tvshow.nfo
        tvshow_nfo_url = urljoin(parent_dir_url, "tvshow.nfo")
        tvshow_info = parse_nfo_data(tvshow_nfo_url)
        with cache_lock:
            tvshow_cache[parent_dir_url] = tvshow_info or False # False代表尝试过但没有

    # 情况 A: 是剧集 (有 tvshow.nfo)
    if tvshow_info:
        # 获取单集nfo (同名nfo)
        ep_nfo_url = url.rsplit('.', 1)[0] + ".nfo"
        ep_info = parse_nfo_data(ep_nfo_url)
        
        title = tvshow_info.get("title") or path_parts[-2]
        year = tvshow_info.get("year") or ""
        tmdb = tvshow_info.get("tmdbid") or "0"
        
        # 格式化 S01E01
        s_e = "S01E01" # 默认
        if ep_info:
            try:
                s = int(ep_info.get("season") or 1)
                e = int(ep_info.get("episode") or 1)
                s_e = f"S{s:02d}E{e:02d}"
            except: pass
        
        new_dir = f"{path_parts[-2]} {{tmdb-{tmdb}}}"
        new_filename = f"{title}.{year}.{s_e}{res_suffix}.strm"
        new_path = "/".join(path_parts[:-2] + [new_dir, new_filename])
        return new_path

    # 情况 B: 是电影 (没有 tvshow.nfo, 寻找同名nfo)
    movie_nfo_url = url.rsplit('.', 1)[0] + ".nfo"
    movie_info = parse_nfo_data(movie_nfo_url)
    
    if movie_info:
        title = movie_info.get("title") or filename.rsplit('.', 1)[0]
        year = movie_info.get("year") or ""
        tmdb = movie_info.get("tmdbid") or "0"
        
        new_dir = f"{path_parts[-2]} {{tmdb-{tmdb}}}"
        new_filename = f"{title}.{year}{res_suffix}.strm"
        new_path = "/".join(path_parts[:-2] + [new_dir, new_filename])
        return new_path

    # 情况 C: 无任何nfo，保持原样
    return decoded_url.replace(BASE_URL.rstrip('/'), "")

def process_url(url):
    session = get_session()
    
    # 1. 处理 .strm 文件
    if url.lower().endswith('.strm'):
        try:
            resp = session.get(url, timeout=10)
            content = resp.text.strip().replace('\n', '').replace('\r', '')
            
            # 核心修改：根据NFO重构路径
            final_path = get_new_strm_path(url, content)

            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{final_path}#{content}\n")
            
            with stats_lock:
                stats["files"] += 1
        except Exception as e:
            # print(f"Error processing {url}: {e}")
            pass
        return

    # 2. 处理目录
    if url.endswith('/') or url == BASE_URL:
        try:
            resp = session.get(url, timeout=10)
            html_content = resp.content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'lxml')
            
            with stats_lock:
                stats["dirs"] += 1

            for link in soup.find_all('a'):
                href = link.get('href')
                if not href or href.startswith('?') or href.startswith('/') or href == '../':
                    continue
                
                full_url = urljoin(url, href)
                is_dir = href.endswith('/')
                is_strm = href.lower().endswith('.strm')
                
                if (is_dir or is_strm) and full_url.startswith(BASE_URL):
                    with visited_lock:
                        if full_url not in visited_urls:
                            visited_urls.add(full_url)
                            task_queue.put(full_url)
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
    # 初始化清空文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print(f"🚀 启动重构爬虫，线程数: {THREAD_COUNT}")
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
                print(f"进度: 目录 {stats['dirs']} | strm {stats['files']} | 缓存NFO {len(tvshow_cache)} | 队列 {task_queue.qsize()}   ", end='\r')
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n停止")

    print(f"\n\n✅ 完成！耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()