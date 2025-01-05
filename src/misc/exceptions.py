"""
Exceptions
==========


"""


class CorruptedFileException(Exception):
    """
    Raised when a file is detected to be corrupted or malformed.

    :param message: Optional error message describing the corruption.
    :type message: str, optional

    **Usage:**

    * Raise this exception when a file cannot be parsed or read due to corruption.
    * Provide a descriptive error message to aid in debugging.

    """

    pass


class UnsupportedFileTypeException(Exception):
    """
    Raised when a file type is not supported by the application or module.

    :param message: Optional error message describing the unsupported file type.
    :type message: str, optional

    **Usage:**

    * Raise this exception when a file with an unsupported type is encountered.
    * Provide a descriptive error message indicating the unsupported file type and possibly suggesting alternatives.

    """

    pass


class ModelServingTypeNotSupported(Exception):
    """
    Raised when a model's serving type is not supported by the application.

    :param message: Optional error message describing the unsupported serving type.
    :type message: str, optional

    **Usage:**

    * Raise this exception when a model's specified serving type is incompatible with the system.
    * Provide a descriptive error message detailing the unsupported serving type and possibly suggesting alternatives.

    """

    pass


class ModelTypeNotSupported(Exception):
    """
    Raised when a model type is not supported by the application.

    :param message: Optional error message describing the unsupported model type.
    :type message: str, optional

    **Usage:**

    * Raise this exception when a model type is incompatible with the system.
    * Provide a descriptive error message detailing the unsupported model type and possibly suggesting alternatives.

    """

    pass


class VectorstoreTypeNotSupported(Exception):
    """
    Raised when a vectorstore type is not supported by the application.

    :param message: Optional error message describing the unsupported vectorstore type.
    :type message: str, optional

    **Usage:**

    * Raise this exception when a specified vectorstore type is incompatible with the system.
    * Provide a descriptive error message detailing the unsupported serving type and possibly suggesting alternatives.

    """

    pass
