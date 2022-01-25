# this is a one line comment
"""
This is a block comment
also known as a multiline comment
"""

# Two ways to run a python file (AKA module)
# 1. directly: python hello_world.py
# 2. indirectly: by importing it from another module
# Example: in main.py import hello_world

# lets say that i have some code that
# i only want to execude when only hello_world.py is
# run directly
# not when it is imported from another module


def main():  # doesnt need to be called main
    print("hello world from a function in hello_world.py")


if __name__ == "__main__":  # this means that it will only be called if it is ran directly
    main()

# __name__ is the string "__main__" when this module was executed directly

print("__name__ in hello_world.py", __name__)
