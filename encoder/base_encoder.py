from abc import *

class BaseEncoder(object, metaclass = ABCMeta):
    @abstractmethod
    def encode(self, imgs):
        pass

    def __call__(self, imgs):
        return self.encode(imgs)