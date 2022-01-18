print("hello world from main() #1")
import hello_world  # wont print anything since __name__ is "hello_world"


print("__name__ in main.py", __name__)


# data types
# int
# float
# str
# list
# tuples (immutable lists)
x = 10
print(x, type(x))


# / div
# // int div
# ** exponent
# you can also import the math module

import math # good practice is that you put all the import statements at the top
print(2**3, math.pow(2,3)) # 2**3 is an int, but math.pow returns a float

# user input
fav_num = int(input("Enter your favorite number: "))
print("fav_num:",fav_num, type(fav_num))

temp = 30
if(temp > 32):
    print("It is not freezing!")
elif(temp < 32):
    print("it is freezing!")
else:
    print("It is exactly freezing!")