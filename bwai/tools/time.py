from time import perf_counter

class TestTime(object):
    def __init__(self, name=None, print_time=True, format='{:.3e}'):
        self.print_time = print_time
        self.format = format
        self.name=name
    
    def __enter__(self):
        self.start = perf_counter()
        
    def __exit__(self, *args, **kwargs):
        self.end = perf_counter()
        self.interval = self.end - self.start
        if self.name is not None:
            print("{} ".format(self.name), end='')
        if self.print_time:
            print("cost time: {}".format(self.format.format(self.interval)))
        return self.interval