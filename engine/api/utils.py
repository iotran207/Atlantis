import random
import string


def randomString(length):
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for i in range(length))
