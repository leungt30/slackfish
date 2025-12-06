import numpy as np
import chess


def fen_to_bitboards(fenstr: str):
    board = chess.Board(fenstr)
    # Mapping piece symbols to chess piece type constants
    piece_type_map = {
        "p": chess.PAWN,
        "n": chess.KNIGHT,
        "b": chess.BISHOP,
        "r": chess.ROOK,
        "q": chess.QUEEN,
        "k": chess.KING,
    }

    piece_order = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
    bitboards = {}

    for piece_symbol in piece_order:
        piece_type = piece_type_map[piece_symbol.lower()]
        color = chess.WHITE if piece_symbol.isupper() else chess.BLACK
        squares = board.pieces(piece_type, color)
        # Iterate directly over squares instead of all 64
        arr = np.zeros(64, dtype=np.uint8)
        for square in squares:
            arr[square] = 1
        bitboards[piece_symbol] = arr

    return bitboards


def piece_count(fen_str: str) -> int:
    score_map = {
        "p": -1,
        "n": -3,
        "b": -3,
        "r": -5,
        "q": -9,
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
    }
    total_score = 0
    for piece in score_map.keys():
        total_score += fen_str.count(piece) * score_map[piece]
    return total_score


def player_turn(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.turn == chess.WHITE else 0


def en_passant_available(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.has_legal_en_passant() else 0


def in_check(fen_str: str) -> int:
    board = chess.Board(fen_str)
    return 1 if board.is_check() else 0


def castling_rights(fen_str: str) -> int:
    board = chess.Board(fen_str)
    white_king_side = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    white_queen_side = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    black_king_side = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    black_queen_side = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    return (white_king_side, white_queen_side, black_king_side, black_queen_side)


def pst_score(fenstr: str) -> int:
    values = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}
    pst = {
        "P": (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            78,
            83,
            86,
            73,
            102,
            82,
            85,
            90,
            7,
            29,
            21,
            44,
            40,
            31,
            44,
            7,
            -17,
            16,
            -2,
            15,
            14,
            0,
            15,
            -13,
            -26,
            3,
            10,
            9,
            6,
            1,
            0,
            -23,
            -22,
            9,
            5,
            -11,
            -10,
            -2,
            3,
            -19,
            -31,
            8,
            -7,
            -37,
            -36,
            -14,
            3,
            -31,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        "N": (
            -66,
            -53,
            -75,
            -75,
            -10,
            -55,
            -58,
            -70,
            -3,
            -6,
            100,
            -36,
            4,
            62,
            -4,
            -14,
            10,
            67,
            1,
            74,
            73,
            27,
            62,
            -2,
            24,
            24,
            45,
            37,
            33,
            41,
            25,
            17,
            -1,
            5,
            31,
            21,
            22,
            35,
            2,
            0,
            -18,
            10,
            13,
            22,
            18,
            15,
            11,
            -14,
            -23,
            -15,
            2,
            0,
            2,
            0,
            -23,
            -20,
            -74,
            -23,
            -26,
            -24,
            -19,
            -35,
            -22,
            -69,
        ),
        "B": (
            -59,
            -78,
            -82,
            -76,
            -23,
            -107,
            -37,
            -50,
            -11,
            20,
            35,
            -42,
            -39,
            31,
            2,
            -22,
            -9,
            39,
            -32,
            41,
            52,
            -10,
            28,
            -14,
            25,
            17,
            20,
            34,
            26,
            25,
            15,
            10,
            13,
            10,
            17,
            23,
            17,
            16,
            0,
            7,
            14,
            25,
            24,
            15,
            8,
            25,
            20,
            15,
            19,
            20,
            11,
            6,
            7,
            6,
            20,
            16,
            -7,
            2,
            -15,
            -12,
            -14,
            -15,
            -10,
            -10,
        ),
        "R": (
            35,
            29,
            33,
            4,
            37,
            33,
            56,
            50,
            55,
            29,
            56,
            67,
            55,
            62,
            34,
            60,
            19,
            35,
            28,
            33,
            45,
            27,
            25,
            15,
            0,
            5,
            16,
            13,
            18,
            -4,
            -9,
            -6,
            -28,
            -35,
            -16,
            -21,
            -13,
            -29,
            -46,
            -30,
            -42,
            -28,
            -42,
            -25,
            -25,
            -35,
            -26,
            -46,
            -53,
            -38,
            -31,
            -26,
            -29,
            -43,
            -44,
            -53,
            -30,
            -24,
            -18,
            5,
            -2,
            -18,
            -31,
            -32,
        ),
        "Q": (
            6,
            1,
            -8,
            -104,
            69,
            24,
            88,
            26,
            14,
            32,
            60,
            -10,
            20,
            76,
            57,
            24,
            -2,
            43,
            32,
            60,
            72,
            63,
            43,
            2,
            1,
            -16,
            22,
            17,
            25,
            20,
            -13,
            -6,
            -14,
            -15,
            -2,
            -5,
            -1,
            -10,
            -20,
            -22,
            -30,
            -6,
            -13,
            -11,
            -16,
            -11,
            -16,
            -27,
            -36,
            -18,
            0,
            -19,
            -15,
            -15,
            -21,
            -38,
            -39,
            -30,
            -31,
            -13,
            -31,
            -36,
            -34,
            -42,
        ),
        "K": (
            4,
            54,
            47,
            -99,
            -99,
            60,
            83,
            -62,
            -32,
            10,
            55,
            56,
            56,
            55,
            10,
            3,
            -62,
            12,
            -57,
            44,
            -67,
            28,
            37,
            -31,
            -55,
            50,
            11,
            -4,
            -19,
            13,
            0,
            -49,
            -55,
            -43,
            -52,
            -28,
            -51,
            -47,
            -8,
            -50,
            -47,
            -42,
            -43,
            -79,
            -64,
            -32,
            -29,
            -32,
            -4,
            3,
            -14,
            -50,
            -57,
            -18,
            13,
            4,
            17,
            30,
            -3,
            -14,
            6,
            -1,
            40,
            18,
        ),
    }
    flip = (
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )
    score = 0
    position = 0
    for char in fenstr.split(" ")[0]:
        if char.upper() in pst.keys():
            # uses position to get value from pst table if white
            # reverse order if black
            # adds piece value and position value from table combined if white
            # subtracts piece value if black
            score += (
                pst[char.upper()][position if char.isupper() else flip[position]]
                + values.get(char.upper())
            ) * (1 if char.isupper() else -1)
            position += 1
        if char.isdigit():
            position += int(char)

    return score


def check_one_move_away(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_check_one_move_away = 0
    black_check_one_move_away = 0

    # Count checks for current turn
    if board.turn == chess.WHITE:
        white_check_one_move_away = sum(
            1 for move in board.legal_moves if board.gives_check(move)
        )
    else:
        black_check_one_move_away = sum(
            1 for move in board.legal_moves if board.gives_check(move)
        )

    # Flip turn and count checks for the other side
    board.turn = not board.turn
    if board.turn == chess.WHITE:
        white_check_one_move_away = sum(
            1 for move in board.legal_moves if board.gives_check(move)
        )
    else:
        black_check_one_move_away = sum(
            1 for move in board.legal_moves if board.gives_check(move)
        )

    return (white_check_one_move_away, black_check_one_move_away)


def legal_moves_per_side(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_moves = 0
    black_moves = 0

    # Count moves for current turn
    if board.turn == chess.WHITE:
        white_moves = sum(1 for _ in board.legal_moves)
    else:
        black_moves = sum(1 for _ in board.legal_moves)

    # Flip turn and count moves for the other side
    board.turn = not board.turn
    if board.turn == chess.WHITE:
        white_moves = sum(1 for _ in board.legal_moves)
    else:
        black_moves = sum(1 for _ in board.legal_moves)

    return (black_moves, white_moves)


def isolated_pawns(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_isolated_pawns = np.zeros(64, dtype=np.uint8)
    black_isolated_pawns = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if is_isolated(board, chess.WHITE, square):
            white_isolated_pawns[square] = 1
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if is_isolated(board, chess.BLACK, square):
            black_isolated_pawns[square] = 1
    return (white_isolated_pawns, black_isolated_pawns)


def is_isolated(board: chess.Board, color: chess.Color, square: int) -> int:
    file = chess.square_file(square)
    pawns = board.pieces(chess.PAWN, color)

    # Check left file (if not on a-file)
    if file > 0:
        left_file_mask = chess.BB_FILES[file - 1]
        if pawns & left_file_mask:
            return 0

    # Check right file (if not on h-file)
    if file < 7:
        right_file_mask = chess.BB_FILES[file + 1]
        if pawns & right_file_mask:
            return 0

    return 1


def double_pawns(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_double_pawns = np.zeros(64, dtype=np.uint8)
    black_double_pawns = np.zeros(64, dtype=np.uint8)

    # Group pawns by file for efficiency
    white_pawns_by_file = {}
    for square in board.pieces(chess.PAWN, chess.WHITE):
        file = chess.square_file(square)
        if file not in white_pawns_by_file:
            white_pawns_by_file[file] = []
        white_pawns_by_file[file].append(square)

    for file, squares in white_pawns_by_file.items():
        if len(squares) > 1:
            for sq in squares:
                white_double_pawns[sq] = 1

    black_pawns_by_file = {}
    for square in board.pieces(chess.PAWN, chess.BLACK):
        file = chess.square_file(square)
        if file not in black_pawns_by_file:
            black_pawns_by_file[file] = []
        black_pawns_by_file[file].append(square)

    for file, squares in black_pawns_by_file.items():
        if len(squares) > 1:
            for sq in squares:
                black_double_pawns[sq] = 1

    return (white_double_pawns, black_double_pawns)


def rook_on_7th_rank(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rook_on_7th_rank = np.zeros(64, dtype=np.uint8)
    black_rook_on_7th_rank = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        if square // 8 == 6:  # 7th rank for white
            white_rook_on_7th_rank[square] = 1
    for square in board.pieces(chess.ROOK, chess.BLACK):
        if square // 8 == 1:  # 2nd rank for black
            black_rook_on_7th_rank[square] = 1
    return (white_rook_on_7th_rank, black_rook_on_7th_rank)


def rook_on_semi_open_file(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rook_on_semi_open_file = np.zeros(64, dtype=np.uint8)
    black_rook_on_semi_open_file = np.zeros(64, dtype=np.uint8)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        if is_semi_open_file(board, chess.WHITE, square):
            white_rook_on_semi_open_file[square] = 1
    for square in board.pieces(chess.ROOK, chess.BLACK):
        if is_semi_open_file(board, chess.BLACK, square):
            black_rook_on_semi_open_file[square] = 1
    return (white_rook_on_semi_open_file, black_rook_on_semi_open_file)


def is_semi_open_file(board: chess.Board, color: chess.Color, square: int) -> int:
    file = chess.square_file(square)
    file_mask = chess.BB_FILES[file]
    pawns_on_file = board.pieces(chess.PAWN, color) & file_mask
    # Check if there are any pawns of this color on this file (excluding the rook square)
    return 0 if pawns_on_file & ~chess.BB_SQUARES[square] else 1


def rooks_on_same_file(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_rooks_on_same_file = np.zeros(64, dtype=np.uint8)
    black_rooks_on_same_file = np.zeros(64, dtype=np.uint8)

    # Group rooks by file for efficiency
    white_rooks_by_file = {}
    for square in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(square)
        if file not in white_rooks_by_file:
            white_rooks_by_file[file] = []
        white_rooks_by_file[file].append(square)

    for file, squares in white_rooks_by_file.items():
        if len(squares) > 1:
            for sq in squares:
                white_rooks_on_same_file[sq] = 1

    black_rooks_by_file = {}
    for square in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(square)
        if file not in black_rooks_by_file:
            black_rooks_by_file[file] = []
        black_rooks_by_file[file].append(square)

    for file, squares in black_rooks_by_file.items():
        if len(squares) > 1:
            for sq in squares:
                black_rooks_on_same_file[sq] = 1

    return (white_rooks_on_same_file, black_rooks_on_same_file)


def is_forking(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    return is_forking_board(board)


def is_forking_board(board: chess.Board) -> tuple:
    white_forking = 0
    black_forking = 0

    # Check white knights
    for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
        attacks = board.attacks(knight_square)
        attacked_pieces = sum(
            1
            for sq in attacks
            if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
        )
        if attacked_pieces > 1:
            white_forking += 1

    # Check black knights
    for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
        attacks = board.attacks(knight_square)
        attacked_pieces = sum(
            1
            for sq in attacks
            if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
        )
        if attacked_pieces > 1:
            black_forking += 1

    return (white_forking, black_forking)


def fork_available(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_fork_available = 0
    black_fork_available = 0

    # Check knights for current turn
    if board.turn == chess.WHITE:
        for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
            attacks = board.attacks(knight_square)
            for target_square in attacks:
                move = chess.Move(knight_square, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    if is_forking_board(board)[0] == 1:
                        white_fork_available += 1
                    board.pop()
    else:
        for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
            attacks = board.attacks(knight_square)
            for target_square in attacks:
                move = chess.Move(knight_square, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    if is_forking_board(board)[1] == 1:
                        black_fork_available += 1
                    board.pop()

    board.turn = not board.turn
    if board.turn == chess.WHITE:
        # Check white knights
        for knight_square in board.pieces(chess.KNIGHT, chess.WHITE):
            attacks = board.attacks(knight_square)
            for target_square in attacks:
                move = chess.Move(knight_square, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    if is_forking_board(board)[0] == 1:
                        white_fork_available += 1
                    board.pop()
    else:
        for knight_square in board.pieces(chess.KNIGHT, chess.BLACK):
            attacks = board.attacks(knight_square)
            for target_square in attacks:
                move = chess.Move(knight_square, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    if is_forking_board(board)[1] == 1:
                        black_fork_available += 1
                    board.pop()

    return (white_fork_available, black_fork_available)


def bishop_activity(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_bishop_activity = 0
    black_bishop_activity = 0
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        attacks = board.attacks(square)
        for target_square in attacks:
            if (
                board.piece_at(target_square) is None
                or board.piece_at(target_square).color == chess.BLACK
            ):
                white_bishop_activity += 1
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        attacks = board.attacks(square)
        for target_square in attacks:
            if (
                board.piece_at(target_square) is None
                or board.piece_at(target_square).color == chess.WHITE
            ):
                black_bishop_activity += 1
    return (white_bishop_activity, black_bishop_activity)


def pinned_pieces(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_pinned_pieces = 0
    black_pinned_pieces = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_pinned(chess.WHITE, square):
                    white_pinned_pieces += 1
            else:
                if board.is_pinned(chess.BLACK, square):
                    black_pinned_pieces += 1
    return (white_pinned_pieces, black_pinned_pieces)


def value_of_hanging_pieces(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_value_of_hanging_pieces = 0
    black_value_of_hanging_pieces = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_attacked_by(
                    chess.BLACK, square
                ) and not board.is_attacked_by(chess.WHITE, square):
                    white_value_of_hanging_pieces += piece_value(piece)
            else:
                if board.is_attacked_by(
                    chess.WHITE, square
                ) and not board.is_attacked_by(chess.BLACK, square):
                    black_value_of_hanging_pieces += piece_value(piece)
    return (white_value_of_hanging_pieces, black_value_of_hanging_pieces)


def piece_value(piece: chess.Piece) -> int:
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type == chess.KNIGHT:
        return 3
    elif piece.piece_type == chess.BISHOP:
        return 3
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9
    elif piece.piece_type == chess.KING:
        return 20000
    return 0


def hanging_pieces_bitboards(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    white_hanging_pieces = np.zeros(64, dtype=np.uint8)
    black_hanging_pieces = np.zeros(64, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                if board.is_attacked_by(
                    chess.BLACK, square
                ) and not board.is_attacked_by(chess.WHITE, square):
                    white_hanging_pieces[square] = 1
            else:
                if board.is_attacked_by(
                    chess.WHITE, square
                ) and not board.is_attacked_by(chess.BLACK, square):
                    black_hanging_pieces[square] = 1
    return (white_hanging_pieces, black_hanging_pieces)


def center_control(fen_str: str) -> tuple:
    board = chess.Board(fen_str)
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    white_center_control = sum(
        sum(1 for _ in board.attackers(chess.WHITE, sq)) for sq in center_squares
    )
    black_center_control = sum(
        sum(1 for _ in board.attackers(chess.BLACK, sq)) for sq in center_squares
    )
    return (white_center_control, black_center_control)


def is_win(fen_str: str) -> int:
    board = chess.Board(fen_str)
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    return 0