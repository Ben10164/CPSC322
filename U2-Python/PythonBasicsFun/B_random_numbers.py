# random numbers
# two main libraries
import random
# also import numpy but we arent using that right now

# we often dont want to have truly random stuff, this is to have reproducable results
# to do this we usually seed the generator
random.seed(0)  # or your favorite number `random.seed(13)`


# is INCLUSIVE, unlike range (so thats weird lol)
die_roll = random.randint(1, 6)
print(die_roll)
