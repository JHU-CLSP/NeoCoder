"""define a crawler abstract class
"""
from abc import ABC, abstractmethod
from typing import Text
from tqdm import tqdm

class Crawler(ABC):
    """
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_info(self):
        """Get extra info from the crawled data.
        """
        raise NotImplementedError("Crawler is an abstract class.")

    @abstractmethod
    def crawl(self):
        """Crawl the data.
        """
        raise NotImplementedError("Crawler is an abstract class.")

    @abstractmethod
    def _crawl(self, url: Text, **kwargs):
        """
        """
        raise NotImplementedError("Crawler is an abstract class.")
    