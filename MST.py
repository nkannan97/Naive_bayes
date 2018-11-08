import numpy as np


def adjacentMatrix(A, u):
    vertex_lst = []
    for x in range(len(A)):
        if A[u][x] > 0 and x != u:
            vertex_lst.insert(0, x)
    vertex_lst = sorted(vertex_lst)
    return vertex_lst


def MinVertex(Q):
    q = Q[0]
    Q.remove(Q[0])
    return q


def maintain_minqueue(Q, K):
    for i in range(len(Q)):
        for j in range(len(Q)):
            if K[Q[i]] > K[Q[j]]:
                s = Q[i]
                Q[i] = Q[j]
                Q[j] = s


def prim(edge_weight_matrix):
    Vertices = list(range(np.shape(edge_weight_matrix)[0]))

    Parents = [None] * len(Vertices)

    Keys = [-float('inf')] * len(Vertices)

    Queue = Vertices

    Keys[0] = 0

    maintain_minqueue(Queue, Keys)

    while len(Queue) > 0:
        u = MinVertex(Queue)
        getVertex = adjacentMatrix(edge_weight_matrix, u)
        for v in getVertex:
            w = edge_weight_matrix[u][v]
            if Queue.count(v) > 0 and w > Keys[v]:
                Parents[v] = u
                Keys[v] = w
                maintain_minqueue(Queue, Keys)
    return Parents
