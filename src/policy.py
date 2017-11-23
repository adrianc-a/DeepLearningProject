
def upper_confidence_bound(node):
    for child in node.children:
        return node.q[child] + (1.0 / (1 + node.n[child]))
    return -1