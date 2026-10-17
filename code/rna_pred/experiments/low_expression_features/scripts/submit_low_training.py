"""Freeze reviewed sources and submit selected training -> numerical/figure summary."""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml


def main():
    project=Path(__file__).resolve().parents[5]
    root=project/"outputs/experiments/low_expression_features"
    with (root/".submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(["squeue","--me","--name=rna-low-train,rna-low-summary","-h","-o","%i"],text=True).strip()
        if active:
            raise RuntimeError(f"Investigation training/summary already active: {active}")
        selection=json.loads((root/"selection.json").read_text())
        if not selection["selected"]:
            raise RuntimeError("No promising screens; report negative findings before choosing further training")
        expected=6*(1+len(selection["selected"]))
        if expected>18 or selection["fits"]!=expected:
            raise ValueError("Selection exceeds or disagrees with the authorized fit budget")
        snapshot=root/"sources"/('training-'+datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        snapshot.mkdir(parents=True)
        for src,dst in [(project/"code/rna_pred",snapshot/"code/rna_pred"),(project/"scripts",snapshot/"scripts"),
                        (project/"tests",snapshot/"tests"),(project/"docs",snapshot/"docs")]:
            shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('README.md','pyproject.toml','environment.yml','environment.resolved.yml'):
            shutil.copy2(project/name,snapshot/name)
        shutil.copy2(root/"selection.json",snapshot/"selection.json")
        config=yaml.safe_load((project/'code/rna_pred/experiments/low_expression_features/configs/default.yaml').read_text())
        config['project_root']=str(project)
        (snapshot/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        from rna_pred.common.config import load_config
        from rna_pred.experiments.low_expression_features.training import materialize_bundles
        materialize_bundles(load_config(snapshot/'config.yaml'),root,project/'outputs/experiments/chromatin_representation/features')
        files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()}
        (snapshot/'sha256.json').write_text(json.dumps(files,indent=2)+'\n')
        (project/'logs').mkdir(exist_ok=True)
        train=subprocess.check_output(['sbatch','--parsable',str(snapshot/'code/rna_pred/experiments/low_expression_features/scripts/euler_low_train.sh'),str(snapshot)],cwd=project,text=True).strip()
        try:
            summary=subprocess.check_output(['sbatch','--parsable','--job-name=rna-low-summary',f'--dependency=afterok:{train}',
                     '--kill-on-invalid-dep=yes',str(snapshot/'code/rna_pred/experiments/low_expression_features/scripts/euler_low_cpu.sh'),str(snapshot),'summarize'],cwd=project,text=True).strip()
        except Exception:
            # Keep the submitted training identity even if dependent submission needs retry.
            (root/'training_submission.json').write_text(json.dumps({'snapshot':str(snapshot),'training_array':train,'summary_job':None},indent=2)+'\n')
            raise
        record={'snapshot':str(snapshot),'training_array':train,'summary_job':summary,'fits':expected,
                'selection_sha256':hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()}
        (root/'training_submission.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
