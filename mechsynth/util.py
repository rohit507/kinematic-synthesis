from public import public

"""
Utility functions and stuff.

Basically the bin for stuff that I don't have a good place to keep. 
"""

@public
def hygienic(decorator):
    """
    Wraps decorators nicely.

    Taken from https://stackoverflow.com/a/6406676/305337
    """
    def new_decorator(original):
        wrapped = decorator(original)
        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
        wrapped.__module__ = original.__module__
        return wrapped
    return new_decorator
