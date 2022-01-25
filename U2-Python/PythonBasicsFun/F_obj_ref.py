# Python is Pass By Reference
def add_one(table):
    for i in range(len(table)):
        for j in range(len(table[i])):
            table[i][j] += 1

def pretty_print(table):
    for row in table:
        for item in row:
            print(item, end=" ")
        print()

matrix = [[0, 1, 2], [3, 4, 5]]

print("before:",matrix)
add_one(matrix)
# The matrix object reference is copied into the table parameter
# thus making table an alies to matrix
print("after:",matrix)


def clear_out_wrong(table):
    table = []
    
def clear_out_right(table):
    table.clear()

clear_out_wrong(matrix)
print("Incorret clear_out:",matrix)
# clear_out_right(matrix)
# print("Correct clear_out:",matrix)


# Shallow vs Deep Copy
# Shallow
matrix_copy = matrix.copy()
print("Shallow copy:",matrix_copy)
matrix_copy[0][0] = "*" 
print("Shallow copy after alteration to the shallow copy:",matrix_copy)
print("Original matrix after alteration to the shallow copy:",matrix)

print()

# Deep
import copy
matrix_copy = copy.deepcopy(matrix)
print("Deep copy:",matrix_copy)
matrix_copy[0][0] = "*"
print("Deep copy after alteration to the deep copy:",matrix_copy)
print("Original matrix after alteration to the deep copy:",matrix)
