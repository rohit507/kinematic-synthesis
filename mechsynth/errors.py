
class NotTermError(Exception):
    """
    The value you tried to add to this context does not inherit from
    mecsynth.term.Term. 
    """
    pass

class NotIdError(Exception):
    """
    Element of the term passed in was not an identifier.
    """
    pass

class IncorrectContextError(Exception):
    """
    Identifier was used with incorrect context.
    """
    pass

class IdentifierDoesNotExistError(Exception):
    """
    The identifier used just isn't in the current context.
    """
    pass

class TermNotInLanguageError(Exception):
    """
    The term you are trying to insert into this context is not in the
    language this algebra is over.
    """
    pass

class IncompatibleTermLanguageError(Exception):
    """
    The language for the context and algebra are incompatible.
    """
    pass

class NotInModelBuildContextError(Exception):
    """
    You are using these functions outside of a model build context, which is
    not allowed.
    """
    pass
