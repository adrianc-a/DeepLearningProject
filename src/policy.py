
def upper_confidence_bound(node, child):
    return node.q[child] + node.u[child]