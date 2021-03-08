import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from graphviz import Digraph

def stgs(logits):
    ''' Straight through gumbel softmax
    '''
    def gs(logits, hard):
        return F.gumbel_softmax(logits, tau=1, hard=hard, eps=1e-10, dim=-1)

    y_soft = gs(logits, False)
    y_hard = gs(logits, True)
    return (y_hard - y_soft).detach() + y_soft

def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot

def m1():
    x = torch.randn((2, 10), requires_grad=True)
    y = torch.tensor([[1,0], [0,1]], dtype=torch.float)
    z = torch.randn([4,3], requires_grad=True)

    l1 = nn.Linear(10, 4, bias=False)
    l2 = nn.Linear(3, 2, bias=False)

    x = l1(x)
    tmp1 = stgs(x)
    tmp2 = torch.matmul(tmp1, z)
    logits = l2(tmp2)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    
    # TODO
    g = make_dot(loss)
    g.view()

    loss.backward()

    print('Done')

m1()
