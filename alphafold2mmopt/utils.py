import time
def cost(f):
    def wrapper(*args,**kwargs):
        begin = time.time()
        r=f(*args,**kwargs)
        print(f.__name__, ":", time.time() - begin, "s")
        return r
    return wrapper