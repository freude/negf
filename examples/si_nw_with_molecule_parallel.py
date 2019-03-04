from mpi4py import MPI
from examples.si_nw_with_molecule import *


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    main(spacing=1.0,
         mol_path='/home/mk/tetracene_dft_wB_pcm_38_32_anion.cube',
         nw_path='./SiNW/SiNW2/',
         eps=3.8,
         comm=comm)
