# Loops
# for item in sequence:
#   body
#

for i in range(5):  # 5 is the stop value, exlusive
    print(i, end=" ")

print()  # just a new line

# range(start, stop, step) [start,stop) += step

for i in range(1, 21):
    print(i * 2, end=" ")

for i in range(2, 40, 2):
    print(i)
print(i + 2)
