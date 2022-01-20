# file IO
# often we want to open a CSV file (comma seperated values)
# and load its contents into a table

def read_table(filename):
    table = []
    # 1. open the file
    infile = open(filename, "r")  # returns an IO/Wrapper
    # 2. process the file
    lines = infile.readlines()  # use if the file is short lol, keeps the `\n`
    for line in lines:
        print(line)  # uses the \n
        print(repr(line))  # how the data is stored, pretty cool
        # strip the newline char
        line = line.strip()  # removes leading and trailing whitespace characters
        # passes in sep, also the numbers become strings
        values = line.split(",")
        print(values)
        # TODO: Convert numeric values
        table.append(values)
    print(lines)
    # 3. close the file
    infile.close()
    return table


table = read_table("data.csv")


def pretty_print(table):
    for row in table:
        for item in row:
            print(item, end=" ")
        print()


print()
pretty_print(table)
