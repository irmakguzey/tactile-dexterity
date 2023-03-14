class NearestNeighborBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.exempted_queue = []

    def put(self, item):
        self.exempted_queue.append(item)
        if len(self.exempted_queue) > self.buffer_size:
            self.exempted_queue.pop(0)

    def get(self):
        item = self.exempted_queue[0]
        self.exempted_queue.pop(0)
        return item

    def choose(self, nn_idxs):
        for idx in range(len(nn_idxs)):
            if nn_idxs[idx] not in self.exempted_queue:
                self.put(nn_idxs[idx])
                return idx

        return len(nn_idxs) - 1