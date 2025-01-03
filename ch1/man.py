class Man:
    def __init__(self, name):
        self.name = name
        print("init complete")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("Dongmin")
m.hello()
m.goodbye()