
def upper_confidence_bound(node, child):
    # this produces a KeyError
    return node.q[child] + node.u[child]
