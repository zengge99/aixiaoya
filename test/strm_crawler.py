import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import threading
import queue
from urllib.parse import urljoin, unquote
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
stats = {"files": 0, "dirs": 0}
stats_lock = threading.Lock()

# 缓存目录探测结果
# 格式: { "dir_url": {"has_tvshow": True/False, "tmdbid": "xxx", "title": "xxx", "year": "xxx"} }
dir_nfo_cache = {}
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
    """提取XML标签内容"""
    node = root.find(tag)
    if node is not None and node.text:
        return node.text.strip()
    if tag in ["tmdbid", "tmdb"]:
        for uid in root.findall('uniqueid'):
            if uid.get('type') == 'tmdb':
                return uid.text.strip()
    return ""

def parse_nfo_data(url):
    """获取并解析NFO"""
    session = get_session()
    try:
        resp = session.get(url, timeout=5)
        if resp.status_code == 200:
            raw_content = resp.content.strip()
            if not raw_content: return None
            root = ET.fromstring(raw_content)
            
            year = get_xml_tag_text(root, "year")
            if not year:
                premiered = get_xml_tag_text(root, "premiered")
                if premiered: year = premiered[:4]
                
            return {
                "title": get_xml_tag_text(root, "title"),
                "year": year,
                "tmdbid": get_xml_tag_text(root, "tmdbid") or get_xml_tag_text(root, "tmdb"),
                "season": get_xml_tag_text(root, "season"),
                "episode": get_xml_tag_text(root, "episode")
            }
    except: pass
    return None

def extract_resolution(path):
    """提取分辨率"""
    match = re.search(r'(2160p|1080p|720p|4k|540p)', path, re.I)
    return match.group(0).lower() if match else ""

def find_tv_root_context(strm_url):
    """向上三级查找tvshow.nfo并返回元数据"""
    decoded_url = unquote(strm_url)
    rel_path = decoded_url.replace(BASE_URL.rstrip('/'), "").strip('/')
    parts = rel_path.split('/')
    
    for i in range(3):
        target_parts_idx = len(parts) - 2 - i
        if target_parts_idx < 0: break
        
        search_url = strm_url.rsplit('/', i + 1)[0] + '/'
        
        with cache_lock:
            if search_url in dir_nfo_cache:
                cached = dir_nfo_cache[search_url]
                if cached["has_tvshow"]:
                    return target_parts_idx, cached
                else: continue 

        nfo_url = urljoin(search_url, "tvshow.nfo")
        data = parse_nfo_data(nfo_url)
        
        if data:
            info = {
                "has_tvshow": True, 
                "tmdbid": data["tmdbid"], 
                "title": data["title"], 
                "year": data["year"]
            }
            with cache_lock:
                dir_nfo_cache[search_url] = info
            return target_parts_idx, info
        else:
            with cache_lock:
                dir_nfo_cache[search_url] = {"has_tvshow": False}
    
    return None, None

def get_reconstructed_path(url):
    """重构路径逻辑"""
    decoded_url = unquote(url)
    rel_path = decoded_url.replace(BASE_URL.rstrip('/'), "").strip('/')
    parts = rel_path.split('/')
    if not parts: return rel_path

    # 1. 探测是否为剧集
    root_idx, tv_info = find_tv_root_context(url)

    if root_idx is not None:
        # --- 剧集逻辑 ---
        ep_nfo_url = url.rsplit('.', 1)[0] + ".nfo"
        ep_data = parse_nfo_data(ep_nfo_url)
        
        # 必须找到季和集，否则保持原名
        s = ep_data.get("season") if ep_data else None
        e = ep_data.get("episode") if ep_data else None
        
        if not s or not e:
            return rel_path # 找不到季/集，直接返回原路径

        try:
            s_e_str = f"S{int(s):02d}E{int(e):02d}"
        except:
            return rel_path

        # 构造文件名: title.年份.S01E01.1080p.strm
        res = extract_resolution(url)
        name_elements = []
        if tv_info.get("title"): name_elements.append(tv_info["title"])
        if tv_info.get("year"): name_elements.append(tv_info["year"])
        name_elements.append(s_e_str)
        if res: name_elements.append(res)
        
        new_filename = ".".join(name_elements) + ".strm"
        
        # 拼接路径并在剧集根目录下级插入 {tmdb-id}
        tmdb_str = f"{{tmdb-{tv_info.get('tmdbid') or '0'}}}"
        new_parts = parts[:root_idx+1] + [tmdb_str] + parts[root_idx+1:-1] + [new_filename]
        return "/".join(new_parts)

    else:
        # --- 电影逻辑 ---
        movie_data = parse_nfo_data(url.rsplit('.', 1)[0] + ".nfo")
        if not movie_data:
            movie_data = parse_nfo_data(url.rsplit('/', 1)[0] + "/movie.nfo")
        
        if movie_data:
            title = movie_data.get("title")
            year = movie_data.get("year")
            res = extract_resolution(url)
            tmdbid = movie_data.get("tmdbid") or "0"
            
            name_elements = []
            if title: name_elements.append(title)
            if year: name_elements.append(year)
            if res: name_elements.append(res)
            
            new_filename = (".".join(name_elements) if name_elements else parts[-1].rsplit('.', 1)[0]) + ".strm"
            
            new_parts = parts[:-1] + [f"{{tmdb-{tmdbid}}}"] + [new_filename]
            return "/".join(new_parts)

    return rel_path

def process_url(url):
    session = get_session()
    if url.lower().endswith('.strm'):
        try:
            resp = session.get(url, timeout=10)
            content = resp.text.strip().replace('\n', '').replace('\r', '')
            new_path = get_reconstructed_path(url)
            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{new_path}#{content}\n")
            with stats_lock:
                stats["files"] += 1
        except: pass
        return

    if url.endswith('/') or url == BASE_URL:
        try:
            resp = session.get(url, timeout=10)
            soup = BeautifulSoup(resp.content.decode('utf-8', errors='ignore'), 'lxml')
            with stats_lock:
                stats["dirs"] += 1
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href or href.startswith('?') or href.startswith('/') or href == '../':
                    continue
                full_url = urljoin(url, href)
                if (href.endswith('/') or href.lower().endswith('.strm')) and full_url.startswith(BASE_URL):
                    with visited_lock:
                        if full_url not in visited_urls:
                            visited_urls.add(full_url)
                            task_queue.put(full_url)
        except: pass

def worker():
    while True:
        try:
            current_url = task_queue.get(timeout=5)
            process_url(current_url)
            task_queue.task_done()
        except queue.Empty: break

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f: pass
    print(f"🚀 启动重构爬虫 (TV 格式优化版), 线程: {THREAD_COUNT}")
    start_time = time.time()
    visited_urls.add(BASE_URL)
    task_queue.put(BASE_URL)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREAD_COUNT)]
    for t in threads: t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with stats_lock:
                print(f"进度: 目录 {stats['dirs']} | strm {stats['files']} | 队列 {task_queue.qsize()}    ", end='\r')
    except KeyboardInterrupt: pass
    print(f"\n\n✅ 完成！")

if __name__ == "__main__":
    main()