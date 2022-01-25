# functions
# start with def (which is a keyword)

def print_even_numbers(stop=40):  # stop will be inclusive
    for i in range(2, stop, 2):
        print(i, end=" ")
    print(i + 2)


print_even_numbers()  # will use default stop of 40
print_even_numbers(20)
print_even_numbers(stop=10)
