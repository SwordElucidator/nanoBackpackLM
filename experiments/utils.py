import heapq


class TopK:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def append(self, value):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, value)
        else:
            if value > self.heap[0]:
                heapq.heappushpop(self.heap, value)

    def top_k(self):
        return sorted(self.heap, reverse=True)

    def __str__(self):
        return self.top_k().__str__()

    def __repr__(self):
        return self.top_k().__repr__()
