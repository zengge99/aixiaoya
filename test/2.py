import requests

url = "你的下载链接"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 1. 检查服务器是否支持 Range
resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
print(f"最终 URL: {resp.url}")
print(f"HTTP 状态码: {resp.status_code}")
print(f"是否支持 Range: {resp.headers.get('Accept-Ranges')}")
print(f"文件内容类型: {resp.headers.get('Content-Type')}")

# 2. 读取前 6 个字节
first_6_bytes = resp.raw.read(6)
print(f"文件头十六进制: {first_6_bytes.hex()}")

if first_6_bytes == b'7z\xbc\xaf\x27\x1c':
    print("确认是标准的 7z 文件头！")
else:
    print("警告：这看起来不是一个 7z 文件。可能是 HTML 页面或被拦截了。")
    print(f"前 6 字节转字符: {first_6_bytes}")