import os
import threading
import queue
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

# --- 配置 ---
BASE_DIR = os.path.abspath(os.getcwd())  # 当前本地目录
OUTPUT_FILE = "local_strm_list.txt"
THREAD_COUNT = 20  # 本地 IO 建议线程数不用像网络请求那么大

# --- 全局变量 ---
task_queue = queue.Queue()
visited_lock = threading.Lock()
file_lock = threading.Lock()
stats = {"files": 0, "dirs": 0, "failed": 0}
stats_lock = threading.Lock()

dir_nfo_cache = {}
cache_lock = threading.Lock()

def get_xml_tag_text(root, tag):
    node = root.find(tag)
    if node is not None and node.text and node.text.strip():
        return node.text.strip()
    
    if tag in ["tmdbid", "tmdb"]:
        for uid in root.findall('uniqueid'):
            if uid.get('type') == 'tmdb' and uid.text:
                return uid.text.strip()
    return ""

def parse_nfo_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read().strip()
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
    match = re.search(r'(2160p|1080p|720p|4k|540p)', path, re.I)
    return match.group(0).lower() if match else ""

def find_tv_root_context(strm_path):
    # 计算相对于根目录的路径
    rel_path = os.path.relpath(strm_path, BASE_DIR)
    parts = rel_path.split(os.sep)
    
    current_dir = os.path.dirname(strm_path)
    
    # 向上查找最多3层
    for i in range(3):
        if current_dir < BASE_DIR: break
        
        with cache_lock:
            if current_dir in dir_nfo_cache:
                cached = dir_nfo_cache[current_dir]
                if cached["has_tvshow"]:
                    # 计算 root_idx: 在 rel_path 中的位置
                    target_parts_idx = len(parts) - 2 - i
                    return target_parts_idx, cached
                else:
                    current_dir = os.path.dirname(current_dir)
                    continue

        nfo_path = os.path.join(current_dir, "tvshow.nfo")
        data = parse_nfo_data(nfo_path)
        
        if data:
            info = {"has_tvshow": True, "tmdbid": data.get("tmdbid"), "title": data.get("title"), "year": data.get("year")}
            with cache_lock:
                dir_nfo_cache[current_dir] = info
            target_parts_idx = len(parts) - 2 - i
            return target_parts_idx, info
        else:
            with cache_lock:
                dir_nfo_cache[current_dir] = {"has_tvshow": False}
        
        current_dir = os.path.dirname(current_dir)
            
    return None, None

def extract_season_episode(filename):
    pattern = r'[Ss](\d+)[Ee](\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_reconstructed_path(file_path):
    rel_path = os.path.relpath(file_path, BASE_DIR)
    parts = rel_path.split(os.sep)
    
    res = extract_resolution(file_path)
    root_idx, tv_info = find_tv_root_context(file_path)

    if root_idx is not None and root_idx >= 0:
        # TV 逻辑
        tmdbid = tv_info.get("tmdbid")
        ep_nfo_path = os.path.splitext(file_path)[0] + ".nfo"
        ep_data = parse_nfo_data(ep_nfo_path)
        s = ep_data.get("season") if ep_data else None
        e = ep_data.get("episode") if ep_data else None

        if not s or not e:
            s, e = extract_season_episode(parts[-1])
        
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

        if tmdbid and s and e:
            # 这里的逻辑是保持原目录结构，但在剧集根目录下插入 {tmdb-id} 文件夹
            new_parts = parts[:root_idx+1] + [f"{{tmdb-{tmdbid}}}"] + parts[root_idx+1:-1] + [new_filename]
            return "/".join(new_parts)
        else:
            with stats_lock: stats["failed"] += 1
            return "刮削失败/" + "/".join(parts[:-1] + [new_filename])
    else:
        # Movie 逻辑
        movie_nfo_path = os.path.splitext(file_path)[0] + ".nfo"
        movie_data = parse_nfo_data(movie_nfo_path)
        if not movie_data:
            movie_data = parse_nfo_data(os.path.join(os.path.dirname(file_path), "movie.nfo"))
        
        if movie_data and movie_data.get("tmdbid"):
            tmdbid = movie_data["tmdbid"]
            title = movie_data.get("title")
            year = movie_data.get("year")
            name_parts = []
            if title: name_parts.append(title)
            if year: name_parts.append(year)
            if res: name_parts.append(res)
            new_filename = (".".join(name_parts) if name_parts else os.path.splitext(parts[-1])[0]) + ".strm"
            new_parts = parts[:-1] + [f"{{tmdb-{tmdbid}}}"] + [new_filename]
            return "/".join(new_parts)
        else:
            with stats_lock: stats["failed"] += 1
            return "刮削失败/" + "/".join(parts)

def process_file(file_path):
    if file_path.lower().endswith('.strm'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip().replace('\n', '').replace('\r', '')
            
            new_path = get_reconstructed_path(file_path)
            
            with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{new_path}#{content}\n")
            with stats_lock:
                stats["files"] += 1
        except Exception as e:
            pass

def worker():
    while True:
        try:
            file_path = task_queue.get(timeout=3)
            process_file(file_path)
            task_queue.task_done()
        except queue.Empty:
            break

def main():
    # 初始化输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f: pass
    
    print(f"🚀 正在扫描本地目录: {BASE_DIR}")
    
    # 遍历本地文件并放入队列
    file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        with stats_lock:
            stats["dirs"] += 1
        for file in files:
            if file.lower().endswith('.strm'):
                file_list.append(os.path.join(root, file))
    
    print(f"📂 扫描到 {len(file_list)} 个 strm 文件，开始处理...")
    
    for f in file_list:
        task_queue.put(f)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREAD_COUNT)]
    for t in threads: t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with stats_lock:
                print(f"进度: 已处理 {stats['files']} | 刮削失败 {stats['failed']} | 待处理 {task_queue.qsize()}    ", end='\r')
    except KeyboardInterrupt:
        print("\n用户终止")

    print(f"\n\n✅ 完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
