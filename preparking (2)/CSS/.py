class Vector:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __add__(self,other):
        return Vector(self.x + other.x , self.y + other.y)
    def __refr__(self):
        return f"x:{self.x},y:{self.y}"
V1=Vector(10,20)
V2=Vector(50,60)
V3=V1+V2
print(V3)
