import time
import threading
import functools
import requests
import psutil
import tracemalloc

# Rate Limiting Decorator
class RateLimit:
    def __init__(self, rate):
        self.rate = rate    
        self.lock = threading.Lock()
        self.calls = 0
        self.last_reset = time.time()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with self.lock:
                if time.time() - self.last_reset > 1:
                    self.calls = 0
                    self.last_reset = time.time()
                if self.calls < self.rate:
                    self.calls += 1
                    return func(*args, **kwargs)
                else:
                    raise Exception("Rate limit exceeded")
        return wrapped

# Connection Pooling
class ConnectionPool:
    def __init__(self, max_size=10):
        self.pool = requests.session()
        self.pool.headers.update({'Connection': 'keep-alive'})
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.pool

    def close(self):
        self.pool.close()

# Smart Caching with TTL
class SmartCache:
    def __init__(self, ttl):
        self.cache = {}
        self.ttl = ttl

    def get(self, key):
        item = self.cache.get(key)
        if item and (time.time() - item['timestamp'] < self.ttl):
            return item['value']
        return None

    def set(self, key, value):
        self.cache[key] = {'value': value, 'timestamp': time.time()}

# Memory Profiling Utilities
class MemoryProfiler:
    @staticmethod
    def start():
        tracemalloc.start()

    @staticmethod
    def snapshot():
        return tracemalloc.take_snapshot()

    @staticmethod
    def display_top(snapshot, limit=10):
        top_stats = snapshot.statistics("lineno")[:limit]
        print("Top memory usage:")
        for stat in top_stats:
            print(stat)

    @staticmethod
    def display_current_memory_usage():
        process = psutil.Process()  
        print(f'Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB')

# Example usage
if __name__ == '__main__':
    MemoryProfiler.start()
    cache = SmartCache(ttl=5)
    cache.set('key1', 'value1')
    print(cache.get('key1'))  # Should print 'value1'
    time.sleep(6)
    print(cache.get('key1'))  # Should print 'None'
    MemoryProfiler.display_current_memory_usage()
