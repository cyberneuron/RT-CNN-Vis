import time

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
