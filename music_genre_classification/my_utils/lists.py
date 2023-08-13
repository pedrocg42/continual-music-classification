from typing import Any


def flatten_list(input_list: [list[Any]]) -> list[Any]:
    flattened_list = []
    for element in input_list:
        flattened_list += element
    return flattened_list
