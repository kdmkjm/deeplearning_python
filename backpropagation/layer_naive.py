class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # dout =  상류에서 넘어온 미분값
    def backward(self, dout):
        # x, y를 서로 바꾼다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
    
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy