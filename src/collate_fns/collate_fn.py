from typing import List, Dict, Tuple, Text, Any, Iterable
from abc import ABC, abstractmethod


class CollateFn(ABC):
    def __init__(
        self,
    ):
        """ Collate function
        """
        super().__init__()
        
    def __call__(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        return self.collate(x)
    
    @abstractmethod
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        raise NotImplementedError("CollateFn is an abstract class.")