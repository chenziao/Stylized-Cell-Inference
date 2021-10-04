from neuron import h
import time

pc = h.ParallelContext()
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())

time.sleep(10.00)
print(MPI_size, MPI_rank)
