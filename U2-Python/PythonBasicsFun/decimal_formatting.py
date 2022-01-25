import math
# Decimal Formatting
# a few ways:
# 1. Old school C-Style
print(math.pi)
print("%.2f" % (math.pi))

# 2. more python way
print("{:.2f}".format(math.pi))

# 3. built in .round() {the best way tbh}
# round is different because it actually returns a number, unlike the format options earlier. that means we can do arithmetic with it
print(round(math.pi, 2))
