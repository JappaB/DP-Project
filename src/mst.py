from itertools import chain
from collections import defaultdict
import operator
import numpy as np

def array_to_dict(arr):
    return {(str(i), str(j)): val for (i, j), val in np.ndenumerate(arr) if j != 0 and i != j}

def mst(dependency_matrix):
    S = array_to_dict(dependency_matrix)
    final_edges = CLE(S, S, {}, defaultdict(list))
    return edges_to_vector(final_edges)

def CLE(S, S_init, best_in, kicks_out):
    max_edges = get_max_edges(S)
    for edge in max_edges:
        best_in[edge[1]] = find_destination(find_origin(edge, S_init), S_init)
    cycle = get_cycle(max_edges)

    if not cycle:
        return max_edges

    S_c, c_node, kicks_out = contract_cycle(S, S_init, cycle, kicks_out)
    CLE(S_c, S_init, best_in, kicks_out)
    new_edges = expand_cycle(c_node, best_in, kicks_out)
    return new_edges

## We need to find which of the contracted nodes is the destination node
## E.g (0, 1_2_3) was actually edge (0, 3)
def find_destination(edge, S_init):
    possible_destinations = edge[1].split('_')
    if len(possible_destinations) == 1:
        return edge
    scores = {}
    for node in possible_destinations:
        scores[(edge[0], node)] = S_init.get(edge[0], node)
    return max(scores.items(), key=operator.itemgetter(1))[0]

## We need to find which of the contracted nodes is the origin node
## E.g (1_2, 3) was actually edge (1, 3)
def find_origin(edge, S_init):
    possible_origins = edge[0].split('_')
    if len(possible_origins) == 1:
        return edge
    scores = {}
    for node in possible_origins:
        scores[(node, edge[1])] = S_init.get(node, edge[1])
    return max(scores.items(), key=operator.itemgetter(1))[0]    

## Return a vector of head-dependent edges
def edges_to_vector(edges):
    heads_vector = np.zeros(len(edges)+1)
    for edge in edges:
        heads_vector[int(edge[1])] = int(edge[0])
    return heads_vector

def expand_cycle(c_node, best_in, kicks_out):
    kicker = best_in[c_node]
    for node, best in best_in.items():
        if best in kicks_out[kicker]:
            best_in[node] = kicker
            # print(kicker, "kicks", best, "of node", node)
    new_edges = set(best_in.values())
    return new_edges

def get_max_edges(S):
    max_edges = []
    nodes = set(k[1] for k in S.keys())
    for node in nodes:
        incoming_edges = [(edge, score) for edge, score in S.items() if edge[1]==node]
        max_edge, _ = max(incoming_edges, key=lambda edge: edge[1])
        max_edges.append(max_edge)
    return max_edges

def get_cycle(edges):
    # convert list of edges to a graph {head: [deps]}
    graph = defaultdict(lambda: [])
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # For each node, return cycle if one is found.
    for start in chain(*edges):
        stack = [start]
        paths = defaultdict(lambda: [])
        paths[start].append(start)
        
        while stack:
            head = stack.pop()
            deps = graph.get(head, [])
            if not deps:
                continue
            
            path = paths[head]
            for dep in deps:
                if dep in path:
                    return path
                dep_path = list(path) + [dep]
                paths[dep] = dep_path
                stack.append(dep)

def contract_cycle(S, S_init, cycle, kicks_out):
    # new edges and scores
    S_c = {}
    # new name for contracted cycle
    c_node = '_'.join(str(n) for n in cycle)
    # total score of cycle
    c_score = sum([S[cycle[i], cycle[(i+1)%len(cycle)]] for i in range(len(cycle))])
    # get cycle edges
    cycle_edges = [e for e in S if set([e[0], e[1]]).issubset(cycle)]
    
    for edge, edge_score in S.items():
        # print(edge, edge_score)
        # if edge in cycle, throw away
        if edge[0] in cycle and edge[1] in cycle:
            continue
        # if edge is outgoing from cycle
        elif edge[0] in cycle:
            if edge_score > S_c.get((c_node, edge[1]), -np.inf):
                S_c[(c_node, edge[1])] = edge_score
        # if incoming to cycle
        elif edge[1] in cycle:
            # incoming edge will kick out cycle edge to same node
            for cycle_edge in cycle_edges:
                if edge not in cycle_edges and cycle_edge[1] == edge[1]:
                    kicker = find_destination(find_origin(edge, S_init), S_init)
                    kicks_out[kicker].append(find_destination(find_origin(cycle_edge, S_init), S_init))
            # score [edge[0], cycle_node] = 
            # max (edge_score - previous cycle edge score + whole cycle score)
            # pred = predecessor of edge[1] in cycle
            pred = cycle[cycle.index(edge[1])-1]
            pred_score = S[pred, edge[1]]
            new_score = edge_score - pred_score + c_score
            if new_score > S_c.get((edge[0], c_node), -np.inf):
               S_c[(edge[0], c_node)] = new_score 
        else:
            S_c[edge] = S[edge]
            
    return S_c, c_node, kicks_out

if __name__=="__main__":
    S_ARR = np.array([[ -1,   9,  10,   9],
                        [ -1,  -1,  20,   3],
                        [ -1,  30,  -1,  30],
                        [ -1,  11,   0,  -1]])
    print("S_TEST gives MST:", MST(S_ARR))