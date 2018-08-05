

class RunningAverage:
    def __init__(self):
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.avg:.4f}'
