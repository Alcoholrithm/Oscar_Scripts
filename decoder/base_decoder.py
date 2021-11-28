from abc import *

class BaseDecoder(object, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, args):
        pass

    def __call__(self, args):
	    return self.decode(args)