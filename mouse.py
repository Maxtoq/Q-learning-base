import numpy as np
import random

def evaluate(qt, map, s, news, a):
    lr = 0.1
    dr = 0.9

    ns = s.y * len(map[0]) + s.x
    nnews = news.y * len(map[0]) + news.x

    qt[ns, a] += lr * (map[news.y, news.x] + dr * np.amax(qt[nnews]) - qt[ns, a])

class Coord:
    def __init__(self):
        self.x = 0
        self.y = 0

    def display(self):
        print("coo", self.x, self.y)

class Mouse:
    def __init__(self):
        self.s = Coord()

    # Take action
    def act(self, a, map, qt):
        news = Coord()
        news.x = self.s.x
        news.y = self.s.y

        if a == 0:
            if self.s.x > 0:
                news.x = self.s.x - 1
        elif a == 1:
            if self.s.x < len(map[0]) - 1:
                news.x = self.s.x + 1
        elif a == 2:
            if self.s.y > 0:
                news.y = self.s.y - 1
        elif a == 3:
            if self.s.y < len(map) - 1:
                news.y = self.s.y + 1

        # Evaluate action
        evaluate(qt, map, self.s, news, a)

        self.s = news


map = np.array([[0, 1, 0],
                [2, -10, 10]])
qt = np.zeros((len(map) * len(map[0]), 4))
e = 1.0
ms = Mouse()

print(qt)
print()
print(map)
print()
print(e, ms.s.x, ms.s.y)

for i in range (0, 100):
    # Choose action
    pol = random.random()
    print(pol)

    if pol > e:
        # Do exploitation
        ez
    else:
        # Do exploration
        a = random.randint(0, 3)

    # Take action
    ms.act(a, map, qt)

    print()
    print("Move:")
    print(qt)
    print()
    print(map)
    print()
    print(e, ms.s.x, ms.s.y)
