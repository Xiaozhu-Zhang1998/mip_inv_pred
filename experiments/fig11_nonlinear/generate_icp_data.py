"""Generate inputs for the standalone nonlinear ICP R experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from invariant_prediction.data import generate_synthetic_data_all_intervention_nonlinear

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--n',type=int,default=100,help='Observations per environment')
parser.add_argument('--trial',type=int,default=2,help='Zero-based trial index')
parser.add_argument('--seed',type=int,default=42,help='Base seed; each trial uses seed+trial')
args=parser.parse_args()
if args.n<=0 or args.trial<0: parser.error('--n must be positive and --trial nonnegative')
S,_,X_list,y_list,n_list=generate_synthetic_data_all_intervention_nonlinear(
    n=args.n,p=6,E=5,p_act=.2,lb=.1,ub=.5,lb_star=.5,ub_star=1,
    sigma_x_min=.5,sigma_x_max=1,sigma_y=.5,p_star_min=2,p_star_max=2,
    random_state=args.seed+args.trial
)
X=np.vstack(X_list)
df=pd.DataFrame(X,columns=[f'X{j}' for j in range(X.shape[1])])
df['Y']=np.concatenate(y_list)
df['E']=np.repeat(np.arange(1,len(n_list)+1),n_list)
out=Path(__file__).resolve().parent/'data'
out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/f'data_{args.n}_{args.trial}.csv',index=False)
pd.DataFrame({'true_invariant':S}).to_csv(out/f'true_{args.n}_{args.trial}.csv',index=False)
print(f'Wrote nonlinear ICP data to {out}')
