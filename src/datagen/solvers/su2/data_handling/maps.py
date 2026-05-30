from ..schemas import SU2_ConvergenceFlag

COLD_MAP = {
            'surface_flow': 'surface_flow.csv',
            'restart_flow': 'flow.dat',
            'flow_csv': 'flow.csv',
            'flow_vtu': 'solution_flow.vtu',
        }

WARM_MAP = {

}

MACHSEQ_MAP = {

}

COMPRESS_WIPE_MAP = {
    SU2_ConvergenceFlag.TIMEOUT: "wipe",
    SU2_ConvergenceFlag.FATAL: "wipe",
    SU2_ConvergenceFlag.DIVERGED: "wipe",
    SU2_ConvergenceFlag.OSCILLATORY: "compress",
    SU2_ConvergenceFlag.STAGNATED: "compress",
    SU2_ConvergenceFlag.ITER_LIMITED: "compress",
    SU2_ConvergenceFlag.CONVERGED: "compress"
}