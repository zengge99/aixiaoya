#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
import re
import requests
import traceback
import json
import sys
import datetime
from urllib.parse import urlparse, quote

# 全局变量
count = 0
failcount = 0
fullscan = True
api_name_cache = {}

def extract_year(path: str):
    """从全路径中提取年份 (1840 - 当前年份+3)"""
    current_year = datetime.datetime.now().year
    max_year = current_year + 3
    parts = path.split("/")
    for part in reversed(parts):
        match = re.search(r'\b(\d{4})\b', part)
        if match:
            try:
                year = int(match.group(1))
                if 1840 < year < max_year:
                    return year
            except ValueError:
                continue
    return None

def get_api_correction(dir_path):
    """从外部 API 获取目录的原始修正名（带缓存）"""
    if dir_path in api_name_cache:
        return api_name_cache[dir_path]
    
    api_url = f"http://127.0.0.1:8889/?q={quote(dir_path)}"
    try:
        resp = requests.get(api_url, timeout=10)
        if resp.status_code == 200:
            raw_name = resp.text.strip()
            if "<html" in raw_name.lower() or "error" in raw_name.lower():
                raw_name = ""
        else:
            raw_name = ""
    except Exception:
        raw_name = ""
    
    api_name_cache[dir_path] = raw_name
    return raw_name

def walk(headers:dict, api_url:str, current_path="/", output_file=None, replaceroot=None, lastpath=None):
    global failcount, fullscan, count
    params = {"path": current_path}
    
    try:
        resp = requests.post(
            api_url + "/api/fs/list",
            headers=headers,
            json=params,
            timeout=(5, 15),
        )
        resp.raise_for_status()
        data_dict = resp.json()
        items = data_dict['data']['content']
        failcount = 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        traceback.print_exc()
        print(f"Request failed to server: {api_url}")
        time.sleep(1)
        failcount += 1
        if failcount > 10:
            raise SystemExit("连续失败次数过多，退出")
        return

    # print(f"目录 {current_path} 文件数: {len(items)}")
    filetype_re = re.compile(r'\.(png|jpg|jpeg|bmp|gif|doc|nfo|flac|mp3|wma|ape|cue|wav|dst|dff|dts|ac3|eac3|txt|db|pdf)$')
    
    for item in items:
        full_path = f"{current_path}/{item['name']}".replace("//", "/")
        is_dir = item.get("is_dir", False)
            
        if is_dir:
            if not fullscan and not full_path in lastpath:
                continue
                
            if full_path == lastpath:
                print(f"找到断点目录: {full_path}, 开始全扫描")
                time.sleep(1)
                fullscan = True
                
            try:
                walk(headers, api_url, full_path, output_file, replaceroot, lastpath)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                continue
        else:
            if not fullscan:
                continue
            
            # 过滤后缀
            if filetype_re.search(full_path) or "BDMV" in full_path:
                continue
            
            size = int(item.get("size", 0))
            if size > 0 and size < 4096:
                continue
                
            # 路径转换
            final_path = full_path
            if replaceroot is not None:
                if replaceroot == "":
                    final_path = "/" + "/".join(full_path.split("/")[2:])
                else:
                    final_path = "/" + replaceroot + "/" + "/".join(full_path.split("/")[2:])
                final_path = final_path.replace("//", "/").replace("\\/", "|")

            # 修正名合成
            raw_correction = get_api_correction(os.path.dirname(final_path))
            year = extract_year(final_path)
            
            if raw_correction or year:
                year_str = f"({year})" if year else ""
                m_name = f"{raw_correction}{year_str}"
            else:
                m_name = ""

            output_line = f"{final_path}\t{size}\t{m_name}"
            print(output_line)
            
            if output_file:
                if "\n" not in output_line:
                    output_file.write(output_line + "\n")
                output_file.flush()

def extract_url_components(url):
    parsed = urlparse(url)
    schema = parsed.scheme or "http"
    hostname = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
    path = parsed.path.rstrip('/') or '/'
    return schema, hostname, path

def get_alist_token(base_url, username, password):
    url = f"{base_url.rstrip('/')}/api/auth/login"
    payload = {
        "username": username,
        "password": password
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get("code") == 200:
            token = res_data["data"]["token"]
            return token
        else:
            print(f"登录失败: {res_data.get('message', '未知错误')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"网络请求出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='AList 目录遍历增强版 V3')
    parser.add_argument('--url', type=str, required=True, help='AList URL')
    parser.add_argument('--user', type=str, default=None, required=False, help='后端alist用户名')
    parser.add_argument('--password', type=str, default=None, required=False, help='后端alist密码')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--lastpath', type=str, default=None, required=False, help='断点路径')
    parser.add_argument('--replaceroot', type=str, default=None, required=False, help='替换根目录名称')
    args = parser.parse_args()

    schema, hostname, path = extract_url_components(args.url)
    last_path_val = args.lastpath
    
    # 自动获取断点
    if last_path_val is None:
        try:
            if os.path.exists(args.output):
                with open(args.output, "r", encoding="utf-8") as f:
                    tmplines = f.readlines()
                    for i in range(len(tmplines)-1, -1, -1):
                        line = tmplines[i].strip()
                        if line:
                            last_path_val = os.path.dirname(line.split("\t")[0])
                            break
        except Exception:
            pass

    global fullscan
    if last_path_val:
        fullscan = False
        last_path_val = last_path_val.rstrip('/')
        print(f"检测到断点: {last_path_val}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
    }
    if args.user and args.password:
        token = get_alist_token(f"{schema}://{hostname}", args.user, args.password)
        if token:
            headers["Authorization"] = token

    output_file = None
    try:
        output_file = open(args.output, mode="a", encoding="utf-8")
        walk(headers, f"{schema}://{hostname}", path, output_file, args.replaceroot, last_path_val)
    except KeyboardInterrupt:
        print("\n[!] 用户中止 (Ctrl+C)，正在安全退出...")
    except Exception:
        traceback.print_exc()
    finally:
        if output_file:
            output_file.close()
        print("[*] 扫描任务已结束。")

if __name__ == '__main__':
    main()