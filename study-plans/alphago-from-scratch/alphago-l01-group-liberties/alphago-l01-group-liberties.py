import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns a tuple of two sorted coordinate lists: group and liberties.
    """
    l = len(board)

    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    color = board[row][col]

    if color == 0:
        return [], [(row, col)]

    stack_state = [(row, col)]
    visited = {(row, col)}

    liberties = set()
    group = []

    while stack_state:
        temp_row, temp_col = stack_state.pop()

        if board[temp_row][temp_col] == color:
            group.append((temp_row, temp_col))

            for mrow, mcol in actions:
                new_row = temp_row + mrow
                new_col = temp_col + mcol

                if 0 <= new_row < l and 0 <= new_col < l:

                    if board[new_row][new_col] == 0:
                        liberties.add((new_row, new_col))

                    elif (
                        board[new_row][new_col] == color
                        and (new_row, new_col) not in visited
                    ):
                        visited.add((new_row, new_col))
                        stack_state.append((new_row, new_col))

    return sorted(group), sorted(liberties)