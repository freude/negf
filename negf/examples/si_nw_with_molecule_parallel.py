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

    negf_config = """

    dephasing:  -0.0001

    ef1:        2.1
    ef2:        2.1
    tempr:      100
    energy:
        start:  2.0
        end:    2.5
        steps:  3000

    basis:   [1.70, 2.20, 20]

    """

    spacings = np.arange(1.0, 28.0)

    for spacing in spacings:
        fields_config = """

        unit_cell:        [[0, 0, 5.50]]

        left_translations:     10
        right_translations:    10

        fields:

            eps:          3.8

            cation:      "/home/mk/tetracene_dft_wB_pcm_38_32_cation.cube"

            angle:       1.13446
            spacing:     {}

            xyz:
                - cation:       [0.0,    0.0,    0.0]

        """.format(spacing)

        main1(str(int(spacing)),
              nw_path='./SiNW/SiNW2/',
              fields_config=fields_config,
              negf_config=negf_config,
              comm=comm)
