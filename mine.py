import math
a,b=3,5
r=6


# formulas
T=1/2*b
C=math.pi*r**2
CU=2*(3*5+5*6+6*3)
CO=math.pi*r*a+math.pi*r*r
Cy=2*math.pi*r*a+2*math.pi*r**2
sp=4/3*math.pi*r**3
hs=(2/3)*math.pi*r**3
aos=a**2


def maxnum(a,b,c,d):
    maxis=a
    if b>maxis:
        maxis=b
    if c>maxis:
        maxis=c 
    if d>maxis:
        maxis=d     
    return maxis
result=maxnum(5,9,36,3)


# printing
print("AREA OF TRIANGLE:",T)
print("AREA OF CIRCLE:",C)
print("SURFACE AREA OF CUBOID:",CU)
print("SURFACE AREA OF CONE:",CO)
print("SURFACE AREA OF CYLINDER:",Cy)
print("VOLUME OF SPHERE",sp)
print("VOLUME OF HEMISPHERE",hs)
print("AREA OF SQUARE:",aos)
print("maximum is:", result)


#palidrone
def palidrone(some):
    return some==some[::-1]
some="racecar"
ans=palidrone(some)
if ans:
    print("IT IS PALIDRONE")
else:
    print("NOT PALIDRONE")

def is_prime(pp):
    for i in range(2,pp):
        if pp%i==0:
            return False
    return True
last=is_prime(5)
print(last)
