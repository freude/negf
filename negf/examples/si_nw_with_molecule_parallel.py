from mpi4py import MPI
from negf.examples.si_nw_with_molecule import *


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # main(spacing=1.0,
    #      mol_path='/home/mk/tetracene_dft_wB_pcm_38_32_anion.cube',
    #      nw_path='./SiNW/SiNW2/',
    #      eps=3.8,
    #      comm=comm)

    fields_config = """

    unit_cell:        [[0, 0, 5.50]]

    left_translations:     3
    right_translations:    3

    fields:

        eps:          3.8

        cation:      "/home/mk/tetracene_dft_wB_pcm_38_32_cation.cube"

        angle:       1.13446
        spacing:     3.0

        xyz:
            - cation:       [-5.0000000000,    0.0000000000,    -5.0000000000]
            - cation:       [5.0000000000,    0.0000000000,    5.0000000000]
    """

    negf_config = """
    ef1:        2.1
    ef2:        2.1
    tempr:      100
    energy:
        start:  2.1
        end:    2.15
        steps:  50

    """

    main1("1", nw_path='./SiNW/SiNW2/', fields_config=fields_config, negf_config=negf_config, comm=comm)
