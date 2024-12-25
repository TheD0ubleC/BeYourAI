import os
from livereload import Server
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 高效监控文件变化的事件处理器
class FastReloadHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"File changed: {event.src_path}")
        self.server.watch(event.src_path)  # 立即触发 livereload 刷新

# 初始化 livereload 服务器
server = Server()

# 监控根目录及所有子目录
root_dir = '.'
server.watch(root_dir)  # 监控所有文件

# 使用 Watchdog 监控文件变化
event_handler = FastReloadHandler(server)
observer = Observer()
observer.schedule(event_handler, root_dir, recursive=True)
observer.start()

try:
    print("Starting livereload server with ultra-fast file monitoring...")
    server.serve(port=3000, host='0.0.0.0', root=root_dir)
except KeyboardInterrupt:
    observer.stop()
observer.join()
