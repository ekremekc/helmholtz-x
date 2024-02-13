def _create_list(my_list, *args, secs=16):
    """
    :param my_list: list containing the first 4 indices
    :param args:
    :param secs: number of sectors
    :return:
    """
    indices = [x - 1 for x in my_list]
    for i in range(2, secs):
        if i == 15:
            indices += [indices[-1] + args[2]]
            indices += [indices[-1] + args[3]]
        else:
            indices += [indices[-1] + args[0]]
            indices += [indices[-1] + args[1]]
    return indices


def physical_groups():
    """

              | y
              |
              |
    x ________. z


    pl-rear   |  1 | 28 | 50 | 77 |  97 | 124 | 144 | 169 |  ===  1 + 27 + 22 + 27 + 20 + 27 + ... + 20 + 25
    pl-right* |  2 |    | 51 |    |  98 |     | 145 |     |
    pl-top    |  3 | 29 | 52 | 78 |  99 | 125 | 146 | 170 |  ===  3 + 26 + 23 + 26 + 21 + 26 + ... + 21 + 24
    pl-left** |  4 | 30 | 53 |    | 100 |     |     |     |
    pl-bottom |  5 | 31 | 54 | 79 | 101 | 126 | 147 | 171 |  ===  5 + 26 + 23 + 25 + 22 + 25 + ... + 21 + 24
    pl-front  |  6 | 32 | 55 | 80 | 102 | 127 | 148 | 172 |  ===  6 + 26 + 23 + 25 + 22 + 25 + ... + 21 + 24
    b1-rear   |  7 | 33 | 56 | 81 | 103 | 128 | 149 | 173 |
    b-right*  |  8 |    | 57 |    | 104 |     | 150 |     |
    b-top     |  9 | 34 | 58 | 82 | 105 | 129 | 151 | 174 |  ===  9 + 25 + 24 + 24 + 23 + 24 + ... + 22 + 23
    b-bottom  | 10 | 35 | 59 | 83 | 106 | 130 | 152 | 175 |  === 10 + 25 + 24 + 24 + 23 + 24 + ... + 22 + 23
    b-front   | 11 | 36 | 60 | 84 | 107 | 131 | 153 | 176 |  === 11 + 25 + 24 + 24 + 23 + 24 + ... + 22 + 23
    pp-rear   | 12 | 37 | 61 | 85 | 108 | 132 | 154 | 177 |
    pp-right* | 13 |    | 62 |    | 109 |     | 155 |     |
    pp-top    | 14 | 38 | 63 | 86 | 110 | 133 | 156 | 178 |  === 14 + 24 + 25 + 23 + 24 + 23 + ... + 23 + 22
    pp-bottom | 15 | 39 | 64 | 87 | 111 | 134 | 157 | 179 |  === 15 + 24 + 25 + 23 + 24 + 23 + ... + 23 + 22
    pp-front  | 16 | 40 | 65 | 88 | 112 | 135 | 158 | 180 |
    fl-rear   | 17 | 41 | 66 | 89 | 113 | 136 | 159 | 181 |  === 17 + 24 + 25 + 23 + 24 + 23 + ... + 23 + 22
    fl-right* | 18 |    | 67 |    | 114 |     | 160 |     |
    fl-top    | 19 | 42 | 68 | 90 | 115 | 137 | 161 | 182 |
    fl-bottom | 20 | 43 | 69 | 91 | 116 | 138 | 162 | 183 |
    fl-front  | 21 | 44 | 70 | 92 | 117 | 139 | 163 | 184 |
    cc-rear   | 22 | 45 | 71 | 93 | 118 | 140 | 164 | 185 | === 22 + 23 + 26 + 22 + 25 + 22 + ... + 24 + 21
    cc-right* | 23 |    | 72 |    | 119 |     | 165 |     |
    cc-top    | 24 | 46 | 73 | 94 | 120 | 141 | 166 | 186 | === 24 + 22 + 27 + 21 + 26 + 21 + ... + 25 + 20
    cc-left** | 25 | 47 | 74 |    | 121 |     |     |     |
    cc-bottom | 26 | 48 | 75 | 95 | 122 | 142 | 167 | 187 | === 26 + 22 + 27 + 20 + 27 + 20 + ... + 25 + 20
    cc-front  | 27 | 49 | 76 | 96 | 123 | 143 | 168 | 188 | === 27 + 22 + 27 + 20 + 27 + 20 + ... + 25 + 20

    """

# __________ plenum
    pl_rear   = _create_list([1, 28, 50, 77], 20, 27, 20, 25)
    # pl_right
    pl_top    = _create_list([3, 29, 52, 78], 21, 26, 21, 24)
    # pl_left
    pl_bottom = _create_list([5, 31, 54, 79], 22, 25, 21, 24)
    pl_front  = _create_list([6, 32, 55, 80], 22, 25, 21, 24)

# __________ burner
    # b_rear
    # b_right
    b_top     = _create_list([9, 34, 58, 82], 23, 24, 22, 23)
    b_bottom  = _create_list([10, 35, 59, 83], 23, 24, 22, 23)
    b_front   = _create_list([11, 36, 60, 84], 23, 24, 22, 23)

# __________ injection
    # pp_rear
    # pp_right
    pp_top    = _create_list([14, 38, 63, 86], 24, 23, 23, 22)
    pp_bottom = _create_list([15, 39, 64, 87], 24, 23, 23, 22)
    # pp_front

# __________ flame
    fl_rear   = _create_list([17, 41, 66, 89], 24, 23, 23, 22)
    # fl_right
    # fl_top
    # fl_bottom
    # fl_front

# __________ combustion chamber
    cc_rear   = _create_list([22, 45, 71, 93], 25, 22, 24, 21)
    # cc_right
    cc_top    = _create_list([24, 46, 73, 94], 26, 21, 25, 20)
    # cc_left
    cc_bottom = _create_list([26, 48, 75, 95], 27, 20, 25, 20)
    cc_front  = _create_list([27, 49, 76, 96], 27, 20, 25, 20)

    b_lateral  = sorted(b_top + b_bottom)
    pp_lateral = sorted(pp_top + pp_bottom)
    cc_rear    = sorted(fl_rear + cc_rear)

    my_dict = {'pl_rear': pl_rear,
               'pl_top': pl_top,
               'pl_bottom': pl_bottom,
               'pl_front': pl_front,
               'b_lateral': b_lateral,
               'b_front': b_front,
               'pp_lateral': pp_lateral,
               'cc_rear': cc_rear,
               'cc_top': cc_top,
               'cc_bottom': cc_bottom,
               'cc_front': cc_front}

    return my_dict


