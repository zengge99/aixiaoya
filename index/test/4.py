import os
import sqlite3
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from xml.etree import ElementTree as ET

# --- 配置 ---
DB_FILE = "strm_files.db"
TXT_FILE = "strm.txt"
PORT = 8899

# --- 数据库初始化 ---

def init_sqlite(txt_path, db_path):
    if os.path.exists(db_path):
        print(f"数据库 {db_path} 已存在，跳过初始化。")
        return

    print("正在构建百万级数据索引，请稍候...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode = OFF")
    c.execute("PRAGMA synchronous = OFF")
    
    # 建立分层表
    c.execute('''CREATE TABLE nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        parent_id INTEGER,
        is_dir BOOLEAN,
        content TEXT
    )''')
    
    # 插入根节点
    c.execute("INSERT INTO nodes (id, name, parent_id, is_dir) VALUES (0, '', -1, 1)")
    
    dir_cache = {"": 0} # 路径 -> ID 缓存
    batch = []
    start_time = time.time()

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "#" not in line: continue
            
            full_path, content = line.split('#', 1)
            parts = full_path.lstrip('/').split('/')
            
            # 处理目录层级
            curr_parent = 0
            curr_path_acc = ""
            for part in parts[:-1]:
                curr_path_acc = f"{curr_path_acc}/{part}" if curr_path_acc else part
                if curr_path_acc not in dir_cache:
                    c.execute("INSERT INTO nodes (name, parent_id, is_dir) VALUES (?, ?, ?)", (part, curr_parent, 1))
                    dir_cache[curr_path_acc] = c.lastrowid
                curr_parent = dir_cache[curr_path_acc]
            
            # 准备文件数据
            batch.append((parts[-1], curr_parent, 0, content))
            if len(batch) >= 50000:
                c.executemany("INSERT INTO nodes (name, parent_id, is_dir, content) VALUES (?, ?, ?, ?)", batch)
                batch = []

    if batch:
        c.executemany("INSERT INTO nodes (name, parent_id, is_dir, content) VALUES (?, ?, ?, ?)", batch)
    
    # 关键：建立复合索引提升查询性能
    c.execute("CREATE INDEX idx_parent_name ON nodes(parent_id, name)")
    conn.commit()
    conn.close()
    print(f"导入完成，耗时: {time.time() - start_time:.2f}s")

# --- WebDAV 核心处理类 ---

class WebDAVHandler(BaseHTTPRequestHandler):
    
    def _get_db(self):
        # 为每个请求建立独立连接（虽然简单，但能避免多线程问题）
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Allow", "GET, OPTIONS, PROPFIND")
        self.send_header("DAV", "1")  # 告诉客户端这是 WebDAV Class 1
        self.end_headers()

    def _find_node(self, conn, path):
        """根据 URL 路径在数据库中逐层定位节点"""
        path = urllib.parse.unquote(path).strip("/")
        if not path:
            return {"id": 0, "name": "", "is_dir": 1, "content": None}
        
        parts = path.split("/")
        curr_parent = 0
        node = None
        for part in parts:
            cursor = conn.execute("SELECT * FROM nodes WHERE parent_id = ? AND name = ?", (curr_parent, part))
            node = cursor.fetchone()
            if not node: return None
            curr_parent = node['id']
        return node

    def do_GET(self):
        """处理 strm 文件内容的读取"""
        conn = self._get_db()
        node = self._find_node(conn, self.path)
        conn.close()

        if node and not node['is_dir']:
            content = node['content'].encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Not Found")

    def do_PROPFIND(self):
        """核心方法：向客户端提供目录列表 XML"""
        conn = self._get_db()
        node = self._find_node(conn, self.path)
        
        if not node:
            conn.close()
            self.send_error(404)
            return

        depth = self.headers.get("Depth", "1")
        
        # XML 响应头
        self.send_response(207, "Multi-Status")
        self.send_header("Content-Type", "application/xml; charset=utf-8")
        self.end_headers()

        # 构建符合 WebDAV 标准的 XML
        root = ET.Element("D:multistatus", {"xmlns:D": "DAV:"})
        
        def add_response(n, p):
            resp = ET.SubElement(root, "D:response")
            href = ET.SubElement(resp, "D:href")
            # 规范化 URL 路径
            full_url_path = (p.rstrip('/') + '/' + n['name']).replace('//', '/')
            href.text = urllib.parse.quote(full_url_path)
            
            propstat = ET.SubElement(resp, "D:propstat")
            prop = ET.SubElement(propstat, "D:prop")
            
            displayname = ET.SubElement(prop, "D:displayname")
            displayname.text = n['name'] if n['name'] else "/"
            
            resourcetype = ET.SubElement(prop, "D:resourcetype")
            if n['is_dir']:
                ET.SubElement(resourcetype, "D:collection")
            
            getcontentlength = ET.SubElement(prop, "D:getcontentlength")
            getcontentlength.text = str(len(n['content'].encode('utf-8'))) if n['content'] else "0"
            
            status = ET.SubElement(propstat, "D:status")
            status.text = "HTTP/1.1 200 OK"

        # 1. 添加当前节点信息
        add_response(node, self.path)

        # 2. 如果是目录且 Depth 为 1，添加子节点
        if node['is_dir'] and depth == "1":
            cursor = conn.execute("SELECT * FROM nodes WHERE parent_id = ?", (node['id'],))
            for child in cursor:
                add_response(child, self.path)

        conn.close()
        xml_data = ET.tostring(root, encoding="utf-8", method="xml")
        self.wfile.write(b'<?xml version="1.0" encoding="utf-8" ?>\n' + xml_data)

# --- 启动 ---

if __name__ == "__main__":
    # 生成测试数据
    if not os.path.exists(TXT_FILE):
        with open(TXT_FILE, "w", encoding='utf-8') as f:
            f.write("/电影/2023/测试视频.strm#http://example.com/v.mp4\n")
            f.write("/电影/动作/movie1.strm#http://example.com/1.mp4\n")
            f.write("/剧集/电视剧A/E01.strm#http://example.com/e1.mp4\n")

    init_sqlite(TXT_FILE, DB_FILE)

    print(f"正在启动 WebDAV 服务，监听端口: {PORT}...")
    # 使用标准库的 HTTPServer
    server = HTTPServer(("0.0.0.0", PORT), WebDAVHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        server.server_close()
