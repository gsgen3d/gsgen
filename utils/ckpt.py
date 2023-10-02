from pathlib import Path


def get_ckpt_path(ckpt):
    ckpt = Path(ckpt)
    if not ckpt.exists():
        uid, time, day, prompt = str(ckpt).strip().split("|")

        ckpt_dir = Path(f"./checkpoints/{prompt}/{day}/{time}/ckpts/")
        files = ckpt_dir.glob("*.pt")
        try:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
        except:
            return None
        ckpt = latest_file

    return ckpt
