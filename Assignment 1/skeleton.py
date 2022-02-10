import copy
import math

import gym
import random
import requests
import numpy as np
import argparse
import sys
import time
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

# SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["Ch5262li-s"]  # TODO: fill this list with your stil-id's

AI_PIECE = -1
PLAYER_PIECE = 1
WINDOW_COUNT = 4


def call_server(move):
    res = requests.post(SERVER_ADRESS + "move",
                        data={
                            "stil_id": STIL_ID,
                            "move": move,
                            # -1 signals the system to start a new game. any running game is counted as a loss
                            "api_key": API_KEY,
                        })
    # For safety some respose checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()['status']:
        print("Server returned a bad status. Return message: ")
        print(res.json()['msg'])
        exit()
    return res


def check_stats():
    res = requests.post(SERVER_ADRESS + "stats",
                        data={
                            "stil_id": STIL_ID,
                            "api_key": API_KEY,
                        })

    stats = res.json()
    return stats


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env, state):
    env.change_player()  # change to oppoent
    avmoves = available_moves(state)
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarrantee a win
    action = int(input("Select move between 0 - 6: ")) # Play against AI (student)
    #action = student_move(state, 3, -math.inf, math.inf, False)[1] # AI Against AI
    #action = random.choice(list(avmoves)) # Random choice

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


def available_moves(board) -> list:
    moves = []
    for c in range(len(board[0])):
        if board[0][c] == 0:
            moves.append(c)

    return moves


def is_winning_move(board, piece) -> bool:
    # Check horizontal locations for win
    for c in range(len(board[0]) - 3):
        for r in range(len(board)):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(len(board[0])):
        for r in range(len(board) - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(len(board[0]) - 3):
        for r in range(len(board) - 3):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(len(board[0]) - 3):
        for r in range(3, len(board)):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

    return False


def place_piece(board, col, piece):
    for row in range(len(board) - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = piece
            return


# Evaluates players (students) and "AI" position and compares them.
def evaluate(board) -> int:
    player_score = score_count(board, PLAYER_PIECE)
    ai_score = score_count(board, AI_PIECE)
    return player_score - ai_score


def eval_window(window, piece) -> int:
    if window.count(piece) == 3 and window.count(0) == 1:
        return 10
    elif window.count(piece) == 2 and window.count(0) == 2:
        return 3

    return 0


def score_count(board, piece) -> int:
    score = 0

    # It will favor the middle of the board, better odds to win.
    center_array = [int(i) for i in list(board[:, len(board[0])//2])]
    center_count = center_array.count(piece)
    score += center_count * 10

    for r in range(len(board)):  # Horizontal
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(len(board[0]) - 3):
            window = row_array[c:c + WINDOW_COUNT]
            score += eval_window(window, piece)

    for c in range(len(board[0])):  # Vertical
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(len(board) - 3):
            window = col_array[r:r + WINDOW_COUNT]
            score += eval_window(window, piece)

    for r in range(len(board) - 3):  # Positively sloped diagonals
        for c in range(len(board[0]) - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_COUNT)]
            score += eval_window(window, piece)

    for r in range(len(board) - 3):  # Negatively sloped diagonals
        for c in range(len(board[0]) - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_COUNT)]
            score += eval_window(window, piece)

    return score


def is_terminal(board) -> int:
    if is_winning_move(board, PLAYER_PIECE):
        return 1
    elif is_winning_move(board, AI_PIECE):
        return 2
    elif len(available_moves(board)) == 0:
        return 3
    return 0


def student_move(board, depth, alpha, beta, maximizing_player) -> (int, int):
    terminal = is_terminal(board)
    if depth == 0 or terminal != 0:
        match terminal:
            case 1:  # Winning move player (Student)
                return math.inf, None
            case 2:  # Winning move AI (Server/local)
                return -math.inf, None
            case 3:  # Draw
                return 0, None

        # depth == 0, evaluate the position on the board
        return evaluate(board), None

    moves = available_moves(board)  # get legal moves on the board
    candidate = random.choice(moves)  # init candidate, in this case a random valid move
    if maximizing_player:
        value = -math.inf
        for move in moves:
            b_copy = board.copy()  # copy boards current state
            place_piece(b_copy, move, PLAYER_PIECE)
            new_value = student_move(b_copy, depth - 1, alpha, beta, False)[0]

            if new_value > value:  # max:(value, new_value)
                value = new_value
                candidate = move

            alpha = max(alpha, value)
            if value >= beta:
                break

        return value, candidate

    else:
        value = math.inf
        for move in moves:
            b_copy = board.copy()
            place_piece(b_copy, move, AI_PIECE)
            new_value = student_move(b_copy, depth - 1, alpha, beta, True)[0]

            if new_value < value:  # min:(value, new_value)
                value = new_value
                candidate = move

            beta = min(beta, value)
            if value <= alpha:
                break

        return value, candidate


def play_game(vs_server=False):
    """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """
    # default state of the board
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(-1)  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()['msg'])
        botmove = res.json()['botmove']
        state = np.array(res.json()['state'])
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print('You start!')
            print()
        else:
            print('Bot starts!')
            print()

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:
        # Select your move
        t1 = time.time()
        stmove = student_move(state, 6, -math.inf, math.inf, True)[1]
        t2 = time.time()
        print("Student move took " + str(round(t2 - t1, 3)) + " seconds")

        # make both student and bot/server moves
        if vs_server:
            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()['msg'])

            # Extract response values
            result = res.json()['result']
            botmove = res.json()['botmove']
            state = np.array(res.json()['state'])
        else:
            if student_gets_move:
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tied to make an illegal move! Games ends.")
                    break
                state, result, done, _ = env.step(stmove)
                print(state)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env, state)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
            elif result == 0.5:
                print("It's a draw!")
            elif result == -1:
                print("You lost!")
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current gamestate
        print(state)
        print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", help="Play locally", action="store_true")
    group.add_argument("-o", "--online", help="Play online vs server", action="store_true")
    parser.add_argument("-s", "--stats", help="Show your current online stats", action="store_true")
    args = parser.parse_args()

    # Print usage info if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.local:
        play_game(vs_server=False)
    elif args.online:
        play_game(vs_server=True)

    if args.stats:
        stats = check_stats()
        print(stats)

    # TODO: Run program with "--online" when you are ready to play against the server
    # the results of your games there will be logged
    # you can check your stats bu running the program with "--stats"


if __name__ == "__main__":
    main()
