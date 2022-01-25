# Lists
# are like arrays but better
#   mixed types
#   are objects (have methods)
#   grow and shrink

# functions: function(object)
# methods: functions that belong to a specific object: <object>.<method>()


fibs = [1, 1, 2, 3, 5, 8]
print(fibs, type(fibs))

# indexing
print(fibs[2])
for item in fibs:
    print(item, end=" ")
print()
for i in range(len(fibs)):
    print(i, ":", fibs[i])
print()
for i, value in enumerate(fibs):
    print(i, ":", value)
print(fibs[-1])  # python supports negative indexing

# built in functons
print("length:", len(fibs))
print("sum:", sum(fibs))
print("min:", min(fibs))
print("max:", max(fibs))
# does not modify the list, just returns a new one
print("reverse sorted", sorted(fibs, reverse=True))

# list methods
print(fibs)
fibs.append("*")
print(fibs)
# returns the index that the first "*" is located in, returns an error if not there
print(fibs.index("*"))
# fibs.pop(-1)  # removes the value at the index of [-1]
fibs.remove("*")  # removes the first occurence of a "*"
print(fibs)
fibs.sort(reverse=True)  # inplace sort
print(fibs)

# nested lists (AKA 2D lists, AKA tables)
matrix = [[0, 1, 2], [3, 4, 5]]  # goes
print(matrix)
# define and call a pretty_print(table)


def pretty_print(table):
    for row in table:
        for item in row:
            print(item, end=" ")
        print()


pretty_print(matrix)
