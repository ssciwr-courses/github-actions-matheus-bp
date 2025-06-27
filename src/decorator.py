def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Preparing the stage for the function call...")
        func(*args, **kwargs)
        print("Tidying up the messes...")

    return wrapper


@my_decorator
def my_func(a, myprint=True):
    b = a + 2
    if myprint:
        print("b is {}".format(b))


if __name__ == "__main__":
    print("Calling the function")
    my_decorator(my_func(5, myprint=True))
