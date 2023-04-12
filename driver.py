from runners.toy_runner import ToyRunner
from runners.md17_runner import MD17Runner
import argparse
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=str, default="toy",choices=["toy","benzene","ethanol"])
parser.add_argument('--dynamics', type=str, default="score",choices=["score", "force", "both"])

parser.add_argument('--n_epochs_score', type=int, default=320)
parser.add_argument('--n_epochs_force', type=int, default=5000)

parser.add_argument('--hidden_units_score', type=int, default=256)
parser.add_argument('--hidden_units_force', type=int, default=16)

parser.add_argument('--lr_score', type=float, default=1e-5)
parser.add_argument('--lr_force', type=float, default=1e-3)

parser.add_argument('--num_samples', type=int, default=20)
parser.add_argument('--num_generated_samples', type=int, default=10000)

parser.add_argument('--rho', type=float, default=1e-2)

parser.add_argument('--sigma_start', type=float, default=1e0)
parser.add_argument('--sigma_end', type=float, default=1e-2)
parser.add_argument('--n_sigmas', type=int, default=2)

parser.add_argument('--conservative_force', action='store_true')
parser.set_defaults(feature=False)

args = parser.parse_args()

if args.dynamics == "score":
    train_force = False
    score_and_force = False
    save_suffix = "_score_match_ss{:.1e}_se_{:.1e}_ns{:d}".format(args.sigma_start, args.sigma_end, args.n_sigmas)

elif args.dynamics == "force":
    train_force = True
    score_and_force = False
    save_suffix = "_force_match"

elif args.dynamics == "both":
    train_force = False
    score_and_force = True
    save_suffix = "_combined_ss{:.1e}_se_{:.1e}_ns{:d}".format(args.sigma_start, args.sigma_end, args.n_sigmas)


if not os.path.exists("results"):
    os.mkdir("results")


if args.setting == "toy":

    savepath = "results/toy" + save_suffix
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    with open('%s/args_log.txt' % savepath, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    runner = ToyRunner(n_epochs_score=args.n_epochs_score,
        n_epochs_force=args.n_epochs_force,
        rho=args.rho,
        num_generated_samples=args.num_generated_samples,
        hidden_units_score=args.hidden_units_score,
        hidden_units_force=args.hidden_units_force)
    
    runner.train_doublewell(num_samples=args.num_samples,
        conservative_force=args.conservative_force,
        train_force=train_force,
        score_and_force=score_and_force,
        savepath=savepath)


elif args.setting == "benzene" or args.setting == "ethanol":

    savepath = "results/" + args.setting  + save_suffix
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    with open('%s/args_log.txt' % savepath, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    runner = MD17Runner(n_epochs_score=args.n_epochs_score,
        n_epochs_force=args.n_epochs_force,
        rho=args.rho,
        num_generated_samples=args.num_generated_samples)
    
    runner.train_molecule(num_molecules=args.num_samples, molecule_name=args.setting,
        sigma_start=args.sigma_start,sigma_end=args.sigma_end,n_sigmas=args.n_sigmas,
        lr_score=args.lr_score, lr_force=args.lr_force,
        hidden_units_score=args.hidden_units_score, hidden_units_force=args.hidden_units_force,
        train_force=train_force,
        score_and_force=score_and_force,
        savepath=savepath)