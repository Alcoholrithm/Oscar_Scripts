from abc import *

class BaseDecoder(object, metaclass=ABCMeta):
    """
    Base Class to decode Encoded Information.

    """
    @abstractmethod
    def decode(self, args):
        pass

    def __call__(self, args):
	    return self.decode(args)