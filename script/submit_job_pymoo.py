#!/usr/bin/env python
import subprocess
import os
if __name__ == '__main__':

    base_dir = "/home/mogami/bicriteria_pbemo"
    output_dir = f"{base_dir}/output/sh/"           # 絶対パス + 末尾スラッシュ
    os.makedirs(output_dir, exist_ok=True)
    for mult_ref in [1]:
        for roi_type in ["roi-c"]:
            for problem_name in ["DTLZ1", "DTLZ5", "DTLZ6", "DTLZ7"]: #"DTLZ1", "DTLZ3", "DTLZ4", "WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"
                for alg in ['BNSGA2','BIBEA','BSMSEMOA','BNSGA3','BSPEA2','BNSGA2-drs','BSPEA2-drs']: #'BNSGA2', 'BIBEA', 'BSMSEMOA', 'RNSGA2-no','RNSGA2', 'gNSGA2', "IBEA", "SMSEMOA", "NSGA2"
                    if alg == "RNSGA2" and roi_type =="roi-p":
                        continue
                    if alg == "gNSGA2" and roi_type =="roi-c":
                        continue
                    if roi_type == "emo":
                        if alg in ["BNSGA2", "BIBEA", "BSMSEMOA", "RNSGA2", "gNSGA2", "BNSGA2-drs", "BNSGA3", "BSPEA2", "BSPEA2-drs"]:
                            continue
                    if roi_type == "roi-c" or roi_type == "roi-p":
                        if alg in ["NSGA2", "SMSEMOA", "IBEA", "NSGA3","SPEA2"]:
                            continue
                    for n_obj in [2,4,6]:
                        for run_id in range(31):            
                            args = f'--n_obj {n_obj} --problem_name {problem_name} --alg {alg} --run_id {run_id} --roi_type {roi_type} --mult_ref {mult_ref}'
                            outfile = os.path.join(
                                output_dir,
                                f'{problem_name}_{alg}_{roi_type}_m{n_obj}_run{run_id}.out'
                            )
                            subprocess.run([
                                'qsub',
                                '-l', 'walltime=72:00:00',
                                "-j", "oe",                 # stdout/errを統合
                                "-o", outfile,
                                '-F', args,
                                'job_pymoo.sh'
                            ])