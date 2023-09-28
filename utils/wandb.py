import wandb


def get_num_runs(project_name=""):
    try:
        return len(wandb.Api().runs(project_name))
    except ValueError:
        return 0
