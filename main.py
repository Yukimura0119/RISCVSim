import simpy
import random

RANDOM_SEED = 42
LOCAL_MEMORY_SIZE = 4096
GLOBAL_MEMORY_SIZE = 16384

class TileAddress:
    def __init__(self, base, offset, annotation):
        self.base = base # (x, y)
        self.offset = offset # (x, y)
        self.annotation = annotation #e.g. X, Wq, Wk...


class Memory(simpy.Container):
    def __init__(self, env, size, init=0):
        super().__init__(env, capacity=size, init=init)
        self.allocate_list = [] # list of TileAddress objects
        self.size = size

    def allocate(self, amount):
        if amount > self.size:
            pass
        yield self.get(amount)

    def deallocate(self, amount):
        yield self.put(amount)


class RISCVTile(simpy.Resource):
    def __init__(self, env, tile_id):
        super().__init__(env)
        self.tile_id = tile_id
        self.local_memory = simpy.Container(env, LOCAL_MEMORY_SIZE, init=LOCAL_MEMORY_SIZE)

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

    def execute(self, workload):
        yield self.env.timeout(5)
        pass

class RISCVMultiprocessor:
    def __init__(self, env, num_cores):
        self.env = env
        # self.cores = simpy.Resource(env, num_cores)
        self.cores = [RISCVTile(env, i) for i in range(num_cores)]
        self.global_memory = simpy.Container(env, GLOBAL_MEMORY_SIZE, init=GLOBAL_MEMORY_SIZE)

    def softmax(self, data):
        pass

    def matmul(self, data):
        pass

    def execute(self, workload):
        yield self.env.timeout(5)
        pass

def schedule(env, processor, type):
    core_idx = 3
    return core_idx

def workload(env, processor, type):
    # schedule
    core_idx = schedule(env, processor, type)
    core = processor.cores[core_idx]
    with core.request() as request:
        yield request
        print(f"Core {core.tile_id} executing workload of type {type} at time {env.now}")
        yield env.process(core.execute(type))


RANDOM_SEED = 42