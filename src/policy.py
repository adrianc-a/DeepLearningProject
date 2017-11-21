
def upper_confidence_bound(node, move):
    for child in node.children:
        if child.in_move == move:
            return node.q[child] + (1.0 / (1 + node.n[child]))
    return -1