import threading
import cv2
import itertools
import argparse
import time

class StreamReader:
    def __init__(self, src=0,framesleep=None):
        self.framesleep = framesleep
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
    def __enter__(self):
        self.start()
        # print(f"FPS is: {self.cap.get(cv2.CAP_PROP_FPS)}")
        # self.cap.set(cv2.CAP_PROP_FPS, 2)
        return self
    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.framesleep:
                time.sleep(self.framesleep)
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        while True:
            with self.read_lock:
                grabbed = self.grabbed
                frame = None
                if grabbed:
                    frame = self.frame.copy()
            if grabbed:
                yield frame
            else:
                break
        # return grabbed, frame
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.stop()
        self.cap.release()
        print("Exiting capture")
        print(exec_type, exc_value, traceback)
        return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stream', metavar="image", default='http://192.168.16.101:8080/video',
                        help="Image")
    args = parser.parse_args()

    with StreamReader(args.stream) as wc:
        # for i in itertools.count():
        for frame in  wc.read():
            if cv2.waitKey(1) ==27:
                exit(0)
            cv2.imshow("frame",frame)
