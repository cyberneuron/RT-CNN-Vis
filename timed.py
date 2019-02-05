import time
import collections

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result

    return timed

class Timer:
    def __init__(self,name="DefaultTimer",silent=False):
        self.starttime = time.time()
        self.name = name
        self.silent = silent
    def tick(self,message=""):
        if not self.silent:
            now = time.time()
            print('%r: %r:%2.4f s' % (self.name, message, now-self.starttime))
    def reset():
        self.starttime = time.time()
    def pause():
        pass

class FPS:
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0
