import numpy as np
from matplotlib import pyplot as plt
import time

threshold = 100

def detection(mask):
    s = time.time()
    bitmaps = []
    bitmaps_cnt = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1:
                start = x, y
                mask, bitmap, cnt = traversal(mask, start)
                if cnt < threshold:
                    continue
                bitmaps.append(bitmap)
                bitmaps_cnt.append(cnt)
    bitmaps = sorted(list(zip(bitmaps, bitmaps_cnt)), key=lambda x: x[1], reverse=True)
    print('runtime: {}s'.format(time.time()-s))
    
    return bitmaps

def traversal(mask, start):
    bitmap = np.zeros(mask.shape)
    count = 0
    def neighbor(x, y, width, height):
        candidates = [(x+1, y+1), (x-1, y-1), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for c in candidates:
            xp, yp = c
            if xp >= height or yp >= width or xp < 0 or yp < 0:
                candidates.remove(c)
        return candidates

    x, y = start
    h, w = mask.shape[0], mask.shape[1]
    queue = [(x,y)]
    while len(queue) > 0:
        x, y = queue.pop(0)
        bitmap[x,y] = 1
        count += 1
        for c in neighbor(x,y,w,h):
            nx, ny = c
            if mask[nx, ny] == 1:
                mask[nx, ny] = 0
                queue.append((nx, ny))
    return mask, bitmap, count

