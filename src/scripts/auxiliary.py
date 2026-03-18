
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def setup_wandb(cfg_run, cfg_export):
    import wandb

    run = wandb.init(
        project= cfg_export['project_name'],
        name= cfg_export.get('run_name', 'default'),
        config= cfg_run,
        job_type= 'training'
    )

    return run

def avoid_MKL_bug(model_type):

    """
    Takes in the model type and tries to avoid a very specific bug related to Intel, Win, Pytorch, FFT, Complex64. Reverts to Compatible MKL Path if conditions are met.
    """

    import os
    import sys
    import platform

    bugged_model_list = ['FNO_nD'] # List of models that use FFT

    if model_type in bugged_model_list and sys.platform == 'win32' and 'Intel' in platform.processor():

        # Forces MKL to use a strictly compatible math path
        # This is used to avoid a very specific bug that happens when using Intel Cpu on Windows where you are forward propping FFT on Pytorch with Complex64 dtype
        os.environ['MKL_CBWR'] = 'COMPATIBLE'
