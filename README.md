# Generative Modeling by Estimating Gradients of the Data Distribution

This repo builds upon the official implementation for the NeurIPS 2019 paper 
[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) by __Yang Song__ and __Stefano Ermon__. Stanford AI Lab.

Our experiments can replicated by running the file driver.py on the command line. For example, score matching and score-based diffusion in the double well potential setting can be run by:
```
python driver.py --setting "toy" --dynamics "score"
```
The `--setting` flag can also be changed to "ethanol" for the molecular experiment. The ``--dynamics`` flag can be set to "score", "force", or "both" for the three types of diffusion model. To combine score and force matching, run a command such as
```
python driver.py --setting "ethanol" --dynamics "both" --sigma_start 1e0 --sigma_end 1e-2 --n_sigmas 2
```
Above, the sigma flags determine the largest (starting) noise level, the smallest (ending) noise level, and the number of log-linear interpolating levels to include (inclusive of the start and end).


## Dependencies

* ase

* PyTorch Geometric

* bgflow

* PyTorch

* PyYAML

* tqdm

* pillow

* tensorboardX

* seaborn


## References

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

```bib
@inproceedings{song2019sliced,
  author    = {Yang Song and
               Sahaj Garg and
               Jiaxin Shi and
               Stefano Ermon},
  title     = {Sliced Score Matching: {A} Scalable Approach to Density and Score
               Estimation},
  booktitle = {Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2019, Tel Aviv, Israel, July 22-25, 2019},
  pages     = {204},
  year      = {2019},
  url       = {http://auai.org/uai2019/proceedings/papers/204.pdf},
}
```

