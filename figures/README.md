# Figures

Diagnostic output from the trained autoencoder (checkpoint step 148,500).
Regenerate with the script named in each row.

| File | What it shows | Produced by |
|---|---|---|
| `reconstruction_samples.png` | Original / reconstruction / error panels for one frame per lifespan quartile, plus an ASCII preview. | `visualize.py` |
| `border_test.png` | Reconstruction F1 as a function of where the seed is placed on the grid. Competence is a bump centred on the training range. | `border_test.py` |
| `traj_cluster_demo.png` | Trajectory retrieval montage: query trajectory beside its latent nearest neighbours, all at a canonical centre placement. | `traj_cluster_demo.py` |

The full seed corpus lives at
[huggingface.co/datasets/themantralab/gol-emergence-pipeline](https://huggingface.co/datasets/themantralab/gol-emergence-pipeline).
