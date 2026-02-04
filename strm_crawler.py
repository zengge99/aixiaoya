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
THREAD_COUNT = 200 

# --- 全局变量 ---
task_queue = queue.Queue()
visited_urls = set()
visited_lock = threading.Lock()
file_lock = threading.Lock()
stats = {"files": 0, "dirs": 0, "failed": 0}
stats_lock = threading.Lock()

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
    # 优先提取直接子节点的标签 (如 <movie><tmdbid>509730</tmdbid>)
    node = root.find(tag)
    if node is not None and node.text and node.text.strip():
        return node.text.strip()
    
    # 如果没找到直接的 tmdbid 标签，查找直接子节点中的 uniqueid
    if tag in ["tmdbid", "tmdb"]:
        for uid in root.findall('uniqueid'):
            if uid.get('type') == 'tmdb' and uid.text:
                return uid.text.strip()
    return ""

def parse_nfo_data(url):
    session = get_session()
    try:
        resp = session.get(url, timeout=5)
        if resp.status_code == 200:
            raw_content = resp.content.strip()
            if not raw_content: return None
            # 解析 XML
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
    match = re.search(r'(2160p|1080p|720p|4k|540p)', path, re.I)
    return match.group(0).lower() if match else ""

def find_tv_root_context(strm_url):
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
            info = {"has_tvshow": True, "tmdbid": data.get("tmdbid"), "title": data.get("title"), "year": data.get("year")}
            with cache_lock:
                dir_nfo_cache[search_url] = info
            return target_parts_idx, info
        else:
            with cache_lock:
                dir_nfo_cache[search_url] = {"has_tvshow": False}
    return None, None

def get_reconstructed_path(url):
    decoded_url = unquote(url)
    rel_path = decoded_url.replace(BASE_URL.rstrip('/'), "").strip('/')
    parts = rel_path.split('/')
    if not parts: return rel_path

    res = extract_resolution(url)
    root_idx, tv_info = find_tv_root_context(url)

    if root_idx is not None:
        # TV 逻辑
        tmdbid = tv_info.get("tmdbid")
        ep_nfo_url = url.rsplit('.', 1)[0] + ".nfo"
        ep_data = parse_nfo_data(ep_nfo_url)
        s = ep_data.get("season") if ep_data else None
        e = ep_data.get("episode") if ep_data else None
        
        if not s or not e:
            new_filename = parts[-1]
        else:
            try:
                s_e_str = f"S{int(s):02d}E{int(e):02d}"
                name_parts = []
                if tv_info.get("title"): name_parts.append(tv_info["title"])
                if tv_info.get("year"): name_parts.append(tv_info["year"])
                name_parts.append(s_e_str)
                if res: name_parts.append(res)
                new_filename = ".".join(name_parts) + ".strm"
            except:
                new_filename = parts[-1]

        if tmdbid:
            new_parts = parts[:root_idx+1] + [f"{{tmdb-{tmdbid}}}"] + parts[root_idx+1:-1] + [new_filename]
            return "/".join(new_parts)
        else:
            with stats_lock: stats["failed"] += 1
            return "刮削失败/" + "/".join(parts[:-1] + [new_filename])
    else:
        # Movie 逻辑
        movie_nfo_url = url.rsplit('.', 1)[0] + ".nfo"
        movie_data = parse_nfo_data(movie_nfo_url)
        if not movie_data:
            movie_data = parse_nfo_data(url.rsplit('/', 1)[0] + "/movie.nfo")
        
        if movie_data and movie_data.get("tmdbid"):
            tmdbid = movie_data["tmdbid"]
            title = movie_data.get("title")
            year = movie_data.get("year")
            name_parts = []
            if title: name_parts.append(title)
            if year: name_parts.append(year)
            if res: name_parts.append(res)
            new_filename = (".".join(name_parts) if name_parts else parts[-1].rsplit('.', 1)[0]) + ".strm"
            new_parts = parts[:-1] + [f"{{tmdb-{tmdbid}}}"] + [new_filename]
            return "/".join(new_parts)
        else:
            with stats_lock: stats["failed"] += 1
            return "刮削失败/" + rel_path

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
    print(f"🚀 启动重构爬虫, 线程: {THREAD_COUNT}")
    start_time = time.time()
    visited_urls.add(BASE_URL)
    task_queue.put(BASE_URL)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREAD_COUNT)]
    for t in threads: t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with stats_lock:
                print(f"进度: 目录 {stats['dirs']} | strm {stats['files']} | 刮削失败 {stats['failed']} | 待处理队列 {task_queue.qsize()}    ", end='\r')
    except KeyboardInterrupt: pass
    print(f"\n\n✅ 完成！")

if __name__ == "__main__":
    main()