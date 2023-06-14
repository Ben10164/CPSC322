import os
import csv

# one required function that is tested with the provided unit test in test_pa1.py
# do not modify this function header or the unit test
def remove_missing_values(table, header, col_name):
    """Makes a new table that is a copy of the table parameter (without modifying the original
    table), but with rows removed from the table that have missing values in the column with
    label col_name
    Args:
        table (list of list): Data in 2D table format
        header (list of str): Column names corresponding to the table (in the same order)
        col_name (str): Represents the name of the column to check for missing values
    Returns:
        list of list: The new table with removed rows
    """
    new_table = []
    for row in table:
        if row[header.index(col_name)] != "": # Checks to see if the value is empty at the index of col_name
            new_table.append(row)
    return new_table

# User made functions:

# Define/call a function that loads the TV shows data into a 2D Python list (AKA table). 
# Remove (and store) the first row of the table, this is the header of the table
def load_tv_show_data(filename): # RETURNS: 2D list, Header list
    """Loads the TV show data from the CSV file into a 2D Python list (AKA table). 
    Args:
        filename (str): The name of the CSV file to load the data from
    Returns:
        2D table list: The table of TV show data
        1D header list: The header of the table
    """
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        table = []
        header = next(reader)
        for row in reader:
            table.append(row)
    return table, header

# Q1: Which TV show has the highest IMDb rating?
def highest_rating_imdb(table,headers):
    """Returns the name of the TV show with the highest IMDb rating
    Args:
       table (list of list): Data in 2D table format
       header (list of str): Column names corresponding to the table (in the same order)
    Returns:
       str: The name of the TV show with the highest IMDb rating
    """
    temp_data = remove_missing_values(table,headers,"IMDb")
    highest_rating = 0
    for row in temp_data:
        if float(row[headers.index("IMDb")]) > highest_rating: # It is a string, so it needs to be converted to a float
            highest_rating = float(row[headers.index("IMDb")])
            highest_rating_show_name = row[headers.index("Title")]
    return highest_rating_show_name

# Q2: Which streaming service hosts the most TV shows?
def most_streaming_service(table,headers):
    """Returns the name of the streaming service that hosts the most TV shows
    Args:
        table (list of list): Data in 2D table format
        header (list of str): Column names corresponding to the table (in the same order)
    Returns:
        str: The name of the streaming service that hosts the most TV shows
    """
    temp_data = remove_missing_values(table,headers,"Netflix")
    temp_data = remove_missing_values(temp_data,headers,"Hulu")
    temp_data = remove_missing_values(temp_data,headers,"Prime Video")
    temp_data = remove_missing_values(temp_data,headers,"Disney+")

    total_netflix = 0
    total_hulu = 0
    total_prime = 0
    total_disney = 0

    for row in temp_data:
        if row[headers.index("Netflix")] == "1":
            total_netflix += 1
        if row[headers.index("Hulu")] == "1":
            total_hulu += 1
        if row[headers.index("Prime Video")] == "1":
            total_prime += 1
        if row[headers.index("Disney+")] == "1":
            total_disney += 1

    if max(total_netflix,total_hulu,total_prime,total_disney) == total_netflix:
        return "Netflix"
    elif max(total_netflix,total_hulu,total_prime,total_disney) == total_hulu:
        return "Hulu"
    elif max(total_netflix,total_hulu,total_prime,total_disney) == total_prime:
        return "Prime Video"
    else:
        return "Disney+"

# Q3: What is the highest IMDb rating a Disney+ show has?
def highest_rating_disney_plus(table,headers):
    """Returns the name of the TV show with the highest IMDb rating
    Args:
        table (list of list): Data in 2D table format
        header (list of str): Column names corresponding to the table (in the same order)
    Returns:
        str: The name of the TV show with the highest IMDb rating on disney plus
    """
    temp_data = remove_missing_values(table,headers,"Disney+")
    temp_data = remove_missing_values(temp_data,headers,"IMDb")
    rating_idx = headers.index("IMDb") 
    name_idx = headers.index("Title")
    highest_rating = 0
    highest_rating_show_name = ""
    for row in temp_data:
        if row[headers.index("Disney+")] == "1" and float(row[rating_idx]) > highest_rating:
            highest_rating = float(row[rating_idx])
            highest_rating_show_name = row[name_idx]
    return highest_rating_show_name

def main():
    """Drives the program
    """
    filename = os.path.join("input_data", "tv_shows.csv")
    print(filename)

    # TODO: your code here
    table, header = load_tv_show_data(filename)
    # print(header)
    # print(table[0][header.index("IMDb")])
    print("The highest rated show on IMDb is",highest_rating_imdb(table,header), "but it should be Breaking Bad... Just saying")
    print("The streaming service that hosts the most shows is",most_streaming_service(table,header))
    print("The highest rated show on IMDb that is a Disney+ show is",highest_rating_disney_plus(table,header))

if __name__ == "__main__":
    main()



