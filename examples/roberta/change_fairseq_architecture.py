import torch
import sys
import os

assert len(sys.argv) > 1, "You need to enter a checkpoint path."
ckpt_path = sys.argv[1]

target_dir = "checkpoints"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

ckpt = torch.load(ckpt_path)
if "cfg" in ckpt:
    old_name = ckpt["cfg"]["model"]._name
    new_name = old_name.replace("roberta", "dec_roberta")
    ckpt["cfg"]["model"]._name = new_name
else:
    assert "args" in ckpt
    old_name = ckpt["args"].arch
    new_name = old_name.replace("roberta", "dec_roberta")
    ckpt["args"].arch = new_name
print("Rename '{}' to {}.".format(old_name, new_name))

target_path = os.path.join(target_dir, "checkpoint.dec.pt")
torch.save(ckpt, target_path)
print("Save checkpoint to {}.".format(target_path))
