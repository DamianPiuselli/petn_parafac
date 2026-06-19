import os
import urllib.request
import urllib.parse
import zipfile
import sys
import socket
import ssl
import time
import hashlib

class RedirectException(Exception):
    def __init__(self, url):
        self.url = url

class CatchRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RedirectException(newurl)

def get_env_proxy():
    for var in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
        val = os.environ.get(var)
        if val:
            return val
    return None

def get_proxy_handler():
    proxy_url = get_env_proxy()
    if proxy_url:
        return urllib.request.ProxyHandler({
            'http': proxy_url,
            'https': proxy_url
        })
    return None

def get_proxy_host_port():
    proxy_url = get_env_proxy()
    if proxy_url:
        try:
            if not proxy_url.startswith(('http://', 'https://')):
                parsed = urllib.parse.urlparse('http://' + proxy_url)
            else:
                parsed = urllib.parse.urlparse(proxy_url)
            host = parsed.hostname
            port = parsed.port or 8080
            return host, port
        except Exception as e:
            print(f"Error parsing proxy from environment: {e}")
    return None, None

def get_redirect_url(url):
    print("Getting redirect URL...")
    # First: Try direct connection
    opener = urllib.request.build_opener(CatchRedirect())
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        opener.open(req)
        return url
    except RedirectException as e:
        return e.url
    except Exception as e_direct:
        print(f"Direct connection failed to get redirect URL: {e_direct}")
        
        # Second: Try local proxy fallback
        proxy_handler = get_proxy_handler()
        if not proxy_handler:
            print("No proxy environment variables found. Skipping proxy redirect attempt.")
            return None

        print("Retrying with proxy from environment...")
        opener_proxy = urllib.request.build_opener(CatchRedirect())
        opener_proxy.add_handler(proxy_handler)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            opener_proxy.open(req)
            return url
        except RedirectException as e:
            return e.url
        except Exception as e_proxy:
            print(f"Proxy connection failed to get redirect URL: {e_proxy}")
            return None

def fetch_proxy_list():
    print("Fetching public proxy list from GitHub...")
    proxy_urls = [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt"
    ]
    proxies = []
    
    # Direct opener (without proxy)
    opener_direct = urllib.request.build_opener()
    
    # Proxy opener fallback
    proxy_handler = get_proxy_handler()
    
    for url in proxy_urls:
        success = False
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with opener_direct.open(req, timeout=10) as resp:
                content = resp.read().decode('utf-8')
                for line in content.splitlines():
                    line = line.strip()
                    if line and line.endswith(":443"):
                        proxies.append(line)
            success = True
        except Exception as e_direct:
            print(f"Direct fetch of proxy list from {url} failed: {e_direct}")
            
        if not success and proxy_handler:
            try:
                print("Retrying proxy list fetch with environment proxy...")
                opener_proxy = urllib.request.build_opener(proxy_handler)
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with opener_proxy.open(req, timeout=10) as resp:
                    content = resp.read().decode('utf-8')
                    for line in content.splitlines():
                        line = line.strip()
                        if line and line.endswith(":443"):
                            proxies.append(line)
            except Exception as e_proxy:
                print(f"Proxy fetch of proxy list from {url} failed: {e_proxy}")
            
    proxies = list(set(proxies))
    print(f"Found {len(proxies)} unique public proxies on port 443.")
    return proxies

def connect_via_chain(local_host, local_port, pub_host, pub_port, target_host, target_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(6.0)
    s.connect((local_host, local_port))
    
    connect_req1 = f"CONNECT {pub_host}:{pub_port} HTTP/1.1\r\nHost: {pub_host}:{pub_port}\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
    s.sendall(connect_req1.encode())
    
    resp1 = b""
    while b"\r\n\r\n" not in resp1:
        chunk = s.recv(1024)
        if not chunk:
            break
        resp1 += chunk
        if len(resp1) > 8192:
            break
            
    if b"200" not in resp1.split(b"\r\n")[0]:
        s.close()
        raise Exception(f"Local proxy rejected CONNECT: {resp1.decode('utf-8', errors='ignore').strip()}")
        
    connect_req2 = f"CONNECT {target_host}:{target_port} HTTP/1.1\r\nHost: {target_host}:{target_port}\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
    s.sendall(connect_req2.encode())
    
    resp2 = b""
    while b"\r\n\r\n" not in resp2:
        chunk = s.recv(1024)
        if not chunk:
            break
        resp2 += chunk
        if len(resp2) > 8192:
            break
            
    if b"200" not in resp2.split(b"\r\n")[0]:
        s.close()
        raise Exception(f"Public proxy rejected CONNECT: {resp2.decode('utf-8', errors='ignore').strip()}")
        
    return s

def test_proxy_chain_to_target(local_host, local_port, pub_host, pub_port, target_host, target_port):
    try:
        s = connect_via_chain(local_host, local_port, pub_host, pub_port, target_host, target_port)
        # Enable default SSL validation to prevent MITM tampering
        context = ssl.create_default_context()
        
        s.settimeout(4.0)
        ssl_sock = context.wrap_socket(s, server_hostname=target_host)
        ssl_sock.close()
        return True
    except Exception:
        return False

def verify_md5(file_path, expected_md5):
    print("Verifying MD5 checksum...")
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calculated = md5.hexdigest()
    print(f"Expected MD5: {expected_md5}")
    print(f"Calculated MD5: {calculated}")
    return calculated == expected_md5

def download_direct(url, dest_path, expected_size=None, expected_md5=None):
    """
    Attempts to download a URL directly using urllib.
    Supports progress reporting, redirects, and resumption.
    Returns True if successful, False otherwise.
    """
    print(f"Attempting direct download of {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    downloaded_size = 0
    if os.path.exists(dest_path):
        downloaded_size = os.path.getsize(dest_path)
        
    if expected_size and downloaded_size >= expected_size:
        print(f"File is already fully downloaded ({downloaded_size} bytes).")
        if expected_md5:
            if verify_md5(dest_path, expected_md5):
                print("MD5 checksum matches successfully.")
                return True
            else:
                print("MD5 checksum mismatch. Deleting and redownloading...")
                try: os.remove(dest_path)
                except: pass
                downloaded_size = 0
        else:
            return True
            
    if downloaded_size > 0:
        headers['Range'] = f"bytes={downloaded_size}-"
        print(f"Resuming download from byte {downloaded_size}...")
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            headers_info = response.info()
            
            content_type = headers_info.get('Content-Type', '')
            if "text/html" in content_type.lower() or "text/plain" in content_type.lower():
                print(f"Direct download warning: response has content-type {content_type}, might be a block page.")
                
            if status == 200:
                mode = 'wb'
                downloaded_size = 0
            elif status == 206:
                mode = 'ab'
            else:
                print(f"Direct download failed: server returned status {status}")
                return False
                
            content_length = headers_info.get('Content-Length')
            if content_length:
                content_length = int(content_length)
            else:
                content_length = 0
                
            total_size = expected_size or (downloaded_size + content_length)
            
            block_size = 1024 * 64
            with open(dest_path, mode) as f:
                curr_downloaded = 0
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    curr_downloaded += len(chunk)
                    total_downloaded_so_far = downloaded_size + curr_downloaded
                    
                    if total_size:
                        percent = total_downloaded_so_far * 100.0 / total_size
                        sys.stdout.write(f"\rDownloading: {percent:3.1f}% [{total_downloaded_so_far}/{total_size} bytes]")
                    else:
                        sys.stdout.write(f"\rDownloading: [{total_downloaded_so_far} bytes]")
                    sys.stdout.flush()
            print()
            
            if expected_md5:
                if verify_md5(dest_path, expected_md5):
                    print("Direct download: MD5 checksum matches successfully.")
                    return True
                else:
                    print("Direct download: MD5 checksum mismatch.")
                    return False
            return True
    except Exception as e:
        print(f"Direct download failed with error: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    print("Unpacking zip archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extraction completed. Files saved to: {dest_dir}")
        os.remove(zip_path)
        print("Removed temporary zip archive.")
    except Exception as e:
        print(f"Error extracting zip archive: {e}")
        sys.exit(1)

def download_lignin():
    figshare_url = "https://ndownloader.figshare.com/files/41018792"
    expected_md5 = "126fce3a469377f45f490c5983203411"
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/chroma/lignin"))
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "lignin_dataset.zip")

    # Try direct download first
    redirect_url = get_redirect_url(figshare_url)
    if redirect_url:
        print("Redirect URL retrieved. Starting direct download attempt...")
        if download_direct(redirect_url, zip_path, 115800763, expected_md5):
            print("Direct download completed successfully!")
            extract_zip(zip_path, dest_dir)
            return
        else:
            print("Direct download failed. Falling back to proxy chaining tunnel...")
    else:
        print("Failed to get redirect URL. Falling back to proxy chaining tunnel...")

    local_host, local_port = get_proxy_host_port()
    if not local_host or not local_port:
        print("Direct download failed and no environment proxy (HTTP_PROXY/HTTPS_PROXY) is set for tunnel fallback.")
        sys.exit(1)

    target_host = "s3q.ait.dtu.dk"
    target_port = 9000

    # 1. Fetch public proxy list
    proxies = fetch_proxy_list()
    if not proxies:
        print("No public proxies found. Exiting.")
        sys.exit(1)

    proxy_index = 0
    total_file_size = 115800763 # Figshare reported size
    
    while True:
        # Check current downloaded size
        if os.path.exists(zip_path):
            downloaded_size = os.path.getsize(zip_path)
        else:
            downloaded_size = 0

        if downloaded_size >= total_file_size:
            print(f"\nFile is fully downloaded ({downloaded_size} bytes).")
            # Verify checksum
            if verify_md5(zip_path, expected_md5):
                print("MD5 checksum matches successfully.")
                break
            else:
                print("MD5 checksum mismatch! Redownloading from scratch...")
                try: os.remove(zip_path)
                except: pass
                downloaded_size = 0

        print(f"\nCurrently downloaded: {downloaded_size} / {total_file_size} bytes.")
        
        # Find next working proxy
        pub_host, pub_port = None, None
        while proxy_index < len(proxies):
            proxy_str = proxies[proxy_index]
            proxy_index += 1
            phost, pport = proxy_str.split(":")
            pport = int(pport)
            print(f"Testing public proxy {proxy_str} against target {target_host}:{target_port}...")
            if test_proxy_chain_to_target(local_host, local_port, phost, pport, target_host, target_port):
                pub_host, pub_port = phost, pport
                print(f"Found working proxy chain: {pub_host}:{pub_port}")
                break
            else:
                print(f"Proxy {proxy_str} failed validation.")
                
        if not pub_host:
            print("No more working proxies available in list. Reloading proxy list...")
            proxies = fetch_proxy_list()
            proxy_index = 0
            if not proxies:
                print("Could not retrieve new proxies. Exiting.")
                sys.exit(1)
            time.sleep(2)
            continue

        connected_sock = None
        ssl_sock = None
        try:
            connected_sock = connect_via_chain(local_host, local_port, pub_host, pub_port, target_host, target_port)
            print("Proxy tunnel established. Fetching fresh redirect URL from Figshare...")
            
            # Fetch fresh redirect URL (signature has 10s expiration window)
            redirect_url = get_redirect_url(figshare_url)
            if not redirect_url:
                raise Exception("Failed to get redirect URL from Figshare")
                
            parsed = urllib.parse.urlparse(redirect_url)
            path = parsed.path
            if parsed.query:
                path += "?" + parsed.query
                
            # Wrap in SSL
            print("Wrapping socket in SSL...")
            context = ssl.create_default_context()
            
            connected_sock.settimeout(15.0)
            ssl_sock = context.wrap_socket(connected_sock, server_hostname=target_host)
            
            # Send GET request with Range header
            print(f"Sending GET request starting from byte {downloaded_size}...")
            range_header = f"Range: bytes={downloaded_size}-\r\n" if downloaded_size > 0 else ""
            get_req = f"GET {path} HTTP/1.1\r\nHost: {target_host}:{target_port}\r\nUser-Agent: Mozilla/5.0\r\n{range_header}Connection: close\r\n\r\n"
            ssl_sock.sendall(get_req.encode())
            
            # Read response headers
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                chunk = ssl_sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > 65536:
                    break
                    
            if b"\r\n\r\n" not in buffer:
                raise Exception("Failed to receive valid HTTP headers")
                
            header_part, body_part = buffer.split(b"\r\n\r\n", 1)
            headers_lines = header_part.split(b"\r\n")
            status_line = headers_lines[0].decode('utf-8', errors='ignore')
            print(f"Server response status: {status_line}")
            
            # If server returned 200 instead of 206, we must write from scratch
            if "200" in status_line:
                mode = "wb"
                downloaded_size = 0
                print("Server returned 200 OK. Overwriting file from byte 0...")
            elif "206" in status_line:
                mode = "ab"
                print("Server returned 206 Partial Content. Appending to file...")
            else:
                raise Exception(f"Server returned non-success response: {status_line}")
                
            # Parse Content-Type
            content_type = ""
            for line in headers_lines[1:]:
                if line.lower().startswith(b"content-type:"):
                    content_type = line.split(b":", 1)[1].strip().decode('utf-8', errors='ignore')
                    break
            print(f"Content-Type received: {content_type}")
            if "text/html" in content_type.lower() or "text/plain" in content_type.lower():
                raise Exception(f"Suspected transparent proxy block page (Content-Type: {content_type})")

            # Parse content-length of this response chunk
            content_length = 0
            for line in headers_lines[1:]:
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
                    break
                    
            print(f"Expect to download: {content_length} bytes in this chunk")

            # Parse total size
            for line in headers_lines[1:]:
                if line.lower().startswith(b"content-range:"):
                    parts = line.split(b"/")
                    if len(parts) > 1:
                        total_file_size = int(parts[1].strip())
                        print(f"Server confirmed total file size (Content-Range): {total_file_size} bytes")
                elif line.lower().startswith(b"content-length:") and downloaded_size == 0:
                    total_file_size = int(line.split(b":", 1)[1].strip())
                    print(f"Server confirmed total file size (Content-Length): {total_file_size} bytes")
            
            # Download body chunk precisely according to Content-Length
            with open(zip_path, mode) as f:
                f.write(body_part)
                curr_downloaded = len(body_part)
                
                remaining = content_length - len(body_part)
                while remaining > 0:
                    chunk_to_read = min(1024 * 64, remaining)
                    chunk = ssl_sock.recv(chunk_to_read)
                    if not chunk:
                        raise Exception("Socket closed prematurely before receiving full content chunk")
                    f.write(chunk)
                    curr_downloaded += len(chunk)
                    remaining -= len(chunk)
                    
                    total_downloaded_so_far = downloaded_size + curr_downloaded
                    percent = total_downloaded_so_far * 100.0 / total_file_size
                    sys.stdout.write(f"\rDownloading: {percent:3.1f}% [{total_downloaded_so_far}/{total_file_size} bytes]")
                    sys.stdout.flush()
            print(f"\nChunk download finished (completed {os.path.getsize(zip_path)} bytes).")
            
        except Exception as e:
            print(f"\nError during download chunk: {e}")
        finally:
            if ssl_sock:
                try:
                    ssl_sock.close()
                except:
                    pass
            if connected_sock:
                try:
                    connected_sock.close()
                except:
                    pass

    extract_zip(zip_path, dest_dir)

if __name__ == "__main__":
    download_lignin()
