# file IO
# often we want to open a CSV file (comma seperated values)
# and load its contents into a table

def convert_to_numeric(values):
    for i in range(len(values)):
        try:
            numeric_value = float(values[i])
            values[i] = numeric_value
        except ValueError:
            print("Could not convert", values[i], "to a number")

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
        convert_to_numeric(values)
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

def write_table(filename, table):
    outfile = open(filename, "w")
    for j in range(len(table)):
        for i in range(len(table[j])-1):
            outfile.write(str(table[j][i]))
            outfile.write(",")
        outfile.write(str(table[j][i+1]))
        if(j != len(table)-1):
            outfile.write("\n")
    outfile.close()

write_table("data_out.csv", table)