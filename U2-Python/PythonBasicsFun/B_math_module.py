import math  # good practice is that you put all the import statements at the top
print(2**3, math.pow(2, 3))  # 2**3 is an int, but math.pow returns a float

# user input
fav_num = int(input("Enter your favorite number: "))
print("fav_num:", fav_num, type(fav_num))

temp = 30
if(temp > 32):
    print("It is not freezing!")
elif(temp < 32):
    print("it is freezing!")
else:
    print("It is exactly freezing!")

# without using the math module
# / div
# // int div
# ** exponent
