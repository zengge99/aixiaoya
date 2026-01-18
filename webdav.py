import os
import sys
import sqlite3
import argparse
import uvicorn
import base64
import secrets
import asyncio
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from xml.etree.ElementTree import Element, SubElement, tostring, register_namespace
from urllib.parse import quote, unquote, urlparse
from datetime import datetime

register_namespace('D', 'DAV:')

CONFIG = {
    "FILE_PATTERN": "",
    "WATCHED_FILES": {},
    "BACKEND_URL": "",
    "USER": "",
    "PASSWORD": "",
    "AUTH_BACKEND_URL": "",
    "DB_FILE": "alist_vfs_compact.db",
    "OBFUSCATE": False
}

class Logger:
    BLUE, GREEN, YELLOW, RED, PURPLE, RESET, BOLD = "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[95m", "\033[0m", "\033[1m"
    @staticmethod
    def log(tag, message, color=""):
        time_str = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{time_str}] [{tag}] {message}{Logger.RESET}")

security = HTTPBasic()
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_ok = secrets.compare_digest(credentials.username, CONFIG["USER"])
    is_pass_ok = secrets.compare_digest(credentials.password, CONFIG["PASSWORD"])
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def safe_b64_encode(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode('utf-8')).decode('utf-8').replace('=', '')

def get_virtual_dir_path(real_dir: str):
    if not CONFIG["OBFUSCATE"]: return real_dir
    parts = real_dir.strip('/').split('/')
    if not parts or parts == ['']: return "/"
    new_parts = [parts[i] if i < 2 else f"b64_{safe_b64_encode(parts[i])}" for i in range(len(parts))]
    return "/" + "/".join(new_parts)

def init_db_from_pattern():
    temp_db = CONFIG["DB_FILE"] + ".tmp"
    if os.path.exists(temp_db): os.remove(temp_db)
    conn = sqlite3.connect(temp_db)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = OFF")
    c.execute("PRAGMA synchronous = OFF")
    c.execute("PRAGMA cache_size = 8000")
    c.execute('CREATE TABLE folders (id INTEGER PRIMARY KEY, v_path TEXT UNIQUE, r_path TEXT)')
    c.execute('CREATE TABLE entries (folder_id INTEGER, v_name TEXT, r_name TEXT, is_dir BOOLEAN, size INTEGER)')
    c.execute('CREATE UNIQUE INDEX idx_entries_unique ON entries(folder_id, v_name, is_dir)')
    c.execute("INSERT INTO folders (v_path, r_path) VALUES ('/', '/')")
    folder_id_cache = {"/": 1}

    def get_folder_id(v_p, r_p):
        v_p = v_p.rstrip('/') if v_p != '/' else '/'
        if v_p in folder_id_cache: return folder_id_cache[v_p]
        c.execute("INSERT OR IGNORE INTO folders (v_path, r_path) VALUES (?, ?)", (v_p, r_p))
        c.execute("SELECT id FROM folders WHERE v_path = ?", (v_p,))
        fid = c.fetchone()[0]
        if len(folder_id_cache) < 40000: folder_id_cache[v_p] = fid
        return fid

    total_count = 0
    matched_files = glob.glob(CONFIG["FILE_PATTERN"])
    try:
        for txt_path in matched_files:
            if CONFIG["DB_FILE"] in txt_path: continue
            file_valid_count = 0
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    segments = line.strip().split('\t')
                    if len(segments) < 2: continue
                    r_full = segments[0] if segments[0].startswith('/') else '/' + segments[0]
                    try: size = int(segments[1])
                    except: continue
                    m_name = segments[2].strip() if len(segments) > 2 else None
                    r_dir, r_file = os.path.dirname(r_full), os.path.basename(r_full)
                    v_dir = get_virtual_dir_path(r_dir)
                    v_file = f"【{m_name}】.{r_file}" if m_name else r_file
                    fid = get_folder_id(v_dir, r_dir)
                    c.execute("INSERT OR IGNORE INTO entries VALUES (?, ?, ?, 0, ?)", (fid, v_file, r_file, size))
                    curr_v, curr_r = v_dir, r_dir
                    while curr_v != "/":
                        vp_v, vp_r = os.path.dirname(curr_v), os.path.dirname(curr_r)
                        vn_v, vn_r = os.path.basename(curr_v), os.path.basename(curr_r)
                        pid = get_folder_id(vp_v, vp_r)
                        c.execute("INSERT OR IGNORE INTO entries VALUES (?, ?, ?, 1, 0)", (pid, vn_v, vn_r))
                        curr_v, curr_r = vp_v, vp_r
                    file_valid_count += 1
            total_count += file_valid_count
            Logger.log("LOAD", f"{os.path.basename(txt_path)}: {file_valid_count} 条记录", Logger.BLUE)
        c.execute("CREATE INDEX idx_f_id ON entries(folder_id)")
        c.execute("CREATE INDEX idx_v_p ON folders(v_path)")
        conn.commit()
        conn.close()
        os.replace(temp_db, CONFIG["DB_FILE"])
        Logger.log("RELOAD", f"Done. Total: {total_count}. Size: {os.path.getsize(CONFIG['DB_FILE'])//1024//1024}MB", Logger.GREEN)
        CONFIG["WATCHED_FILES"] = {f: os.path.getmtime(f) for f in matched_files}
    except Exception as e:
        Logger.log("ERROR", f"Build failed: {e}", Logger.RED)

async def monitor_task():
    while True:
        await asyncio.sleep(3600)
        curr = glob.glob(CONFIG["FILE_PATTERN"])
        if set(curr) != set(CONFIG["WATCHED_FILES"].keys()) or any(os.path.getmtime(f) > CONFIG["WATCHED_FILES"].get(f, 0) for f in curr):
            Logger.log("MONITOR", "Reloading...", Logger.YELLOW)
            await asyncio.get_event_loop().run_in_executor(None, init_db_from_pattern)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitor_task())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

def build_webdav_xml(items):
    ms = Element('{DAV:}multistatus')
    for it in items:
        res = SubElement(ms, '{DAV:}response')
        href = SubElement(res, '{DAV:}href')
        path = it['v_path'].rstrip('/')
        if it['is_dir'] and path != "": path += '/'
        if not path.startswith('/'): path = '/' + path
        href.text = quote(path)
        propstat = SubElement(res, '{DAV:}propstat')
        prop = SubElement(propstat, '{DAV:}prop')
        SubElement(prop, '{DAV:}displayname').text = it['name']
        if it['is_dir']:
            SubElement(SubElement(prop, '{DAV:}resourcetype'), '{DAV:}collection')
        else:
            SubElement(prop, '{DAV:}getcontentlength').text = str(it['size'])
        SubElement(prop, '{DAV:}getlastmodified').text = "Sat, 01 Jan 2024 00:00:00 GMT"
        status_node = SubElement(propstat, '{DAV:}status')
        status_node.text = 'HTTP/1.1 200 OK'
    return tostring(ms, encoding='utf-8')

@app.api_route("/{path:path}", methods=["PROPFIND", "GET", "OPTIONS", "HEAD"])
async def handle_webdav(request: Request, path: str, username: str = Depends(authenticate)):
    v_path = "/" + unquote(path).strip('/')
    if v_path == "//": v_path = "/"
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={"Allow": "OPTIONS, GET, HEAD, PROPFIND", "DAV": "1, 2"})

    conn = sqlite3.connect(CONFIG["DB_FILE"])
    cursor = conn.cursor()

    if request.method in ["GET", "HEAD"]:
        v_dir, v_file = os.path.dirname(v_path), os.path.basename(v_path)
        v_dir_query = v_dir.rstrip('/') if v_dir != '/' else '/'
        cursor.execute('''SELECT f.r_path, e.r_name FROM entries e 
                          JOIN folders f ON e.folder_id = f.id 
                          WHERE f.v_path = ? AND e.v_name = ? AND e.is_dir = 0''', (v_dir_query, v_file))
        row = cursor.fetchone()
        conn.close()
        if row:
            real_full_path = os.path.join(row[0], row[1]).replace('\\', '/').lstrip('/')
            target_url = f"{CONFIG['AUTH_BACKEND_URL'].rstrip('/')}/{quote(real_full_path)}"
            Logger.log("GET", f"Redirect -> {real_full_path}", Logger.GREEN)
            return RedirectResponse(url=target_url, status_code=302)
        return Response(status_code=404)

    if request.method == "PROPFIND":
        depth = request.headers.get("Depth", "1")
        v_path_query = v_path.rstrip('/') if v_path != '/' else '/'
        items = []

        # 核心逻辑：先判断目标是 目录 还是 文件
        cursor.execute("SELECT id, r_path FROM folders WHERE v_path = ?", (v_path_query,))
        folder_row = cursor.fetchone()

        if folder_row:
            # A. 目标是目录
            fid, r_path_log = folder_row
            items.append({'v_path': v_path_query, 'name': os.path.basename(v_path_query) or "/", 'is_dir': True, 'size': 0})
            if depth == "1":
                cursor.execute("SELECT v_name, is_dir, size FROM entries WHERE folder_id = ?", (fid,))
                for r in cursor.fetchall():
                    child_v = (v_path_query.rstrip('/') + '/' + r[0]).replace('//', '/')
                    items.append({'v_path': child_v, 'name': r[0], 'is_dir': r[1], 'size': r[2]})
            Logger.log("SCAN", f"[Dir] {r_path_log} ({len(items)} items)", Logger.BLUE)
        else:
            # B. 目标可能是文件 (Depth 0 常见)
            v_parent = os.path.dirname(v_path_query)
            v_name = os.path.basename(v_path_query)
            v_parent_query = v_parent.rstrip('/') if v_parent != '/' else '/'
            
            cursor.execute('''SELECT e.v_name, e.is_dir, e.size, f.r_path, e.r_name FROM entries e 
                              JOIN folders f ON e.folder_id = f.id 
                              WHERE f.v_path = ? AND e.v_name = ?''', (v_parent_query, v_name))
            entry_row = cursor.fetchone()
            if entry_row:
                items.append({'v_path': v_path_query, 'name': entry_row[0], 'is_dir': entry_row[1], 'size': entry_row[2]})
                Logger.log("SCAN", f"[File] {entry_row[3]}/{entry_row[4]}", Logger.BLUE)
            else:
                conn.close()
                Logger.log("404", f"Not Found in DB: {v_path_query}", Logger.RED)
                return Response(status_code=404)

        conn.close()
        return Response(content=build_webdav_xml(items), status_code=207, media_type="application/xml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--obfuscate", action="store_true")
    args = parser.parse_args()
    CONFIG.update({"FILE_PATTERN": args.file, "USER": args.user, "PASSWORD": args.password, "OBFUSCATE": args.obfuscate})
    up = urlparse(args.url)
    CONFIG["AUTH_BACKEND_URL"] = f"{up.scheme}://{quote(args.user)}:{quote(args.password)}@{up.netloc}{up.path}"
    init_db_from_pattern()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")
