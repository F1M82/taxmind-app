import time
import random
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import Optional
from loguru import logger
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class RateLimiter:
    def __init__(self, max_per_hour: int, min_delay: float):
        self.min_delay = min_delay
        self.interval = 3600.0 / max_per_hour
        self._last_request = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_request
        wait_time = max(self.min_delay, self.interval - elapsed)
        jitter = random.uniform(0.5, 1.5)
        total_wait = wait_time * jitter
        if total_wait > 0:
            time.sleep(total_wait)
        self._last_request = time.time()

class RobotsTxtCache:
    def __init__(self):
        self._cache = {}

    def is_allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            rp = RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self._cache[domain] = rp
            except Exception:
                return True
        return self._cache[domain].can_fetch(user_agent, url)

class EthicalHttpClient:
    def __init__(self, user_agent: str, min_delay: float = 3.0, max_per_hour: int = 200, respect_robots: bool = True):
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.rate_limiter = RateLimiter(max_per_hour, min_delay)
        self.robots_cache = RobotsTxtCache()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/pdf,*/*",
            "Accept-Language": "en-IN,en;q=0.9",
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30), retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)))
    def get(self, url: str, stream: bool = False, timeout: int = 30) -> Optional[requests.Response]:
        if self.respect_robots and not self.robots_cache.is_allowed(url, self.user_agent):
            logger.warning(f"Skipping (robots.txt): {url}")
            return None
        self.rate_limiter.wait()
        try:
            logger.info(f"GET {url}")
            response = self.session.get(url, stream=stream, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)
                raise
            elif e.response.status_code in (403, 404):
                return None
            raise

    def get_pdf(self, url: str, save_path: str) -> bool:
        response = self.get(url, stream=True)
        if not response:
            return False
        try:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")
            return False
