from logger_config import logger


def some_function():
    logger.info("This is an info message from another file.")
    logger.error("This is an error message from another file.")


if __name__ == "__main__":
    print('calling logger')
    some_function()

    ##Comment
