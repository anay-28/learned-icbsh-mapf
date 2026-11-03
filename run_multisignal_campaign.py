"""Local bounded campaign; stops on stage failure, logs actual progress."""
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path('results/multisignal_v1')

def main():
    metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                  platform=platform.platform(),pid=os.getpid(),workers=2,
                  git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (ROOT/'campaign_environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for module,args in [('multisignal_study',['collect','--workers','2']),
                            ('multisignal_evaluate',['fit']),
                            ('multisignal_evaluate',['validate']),
                            ('multisignal_evaluate',['benchmark'])]:
            stage=args[0]
            with (ROOT/(stage+'.campaign.log')).open('x',encoding='utf-8') as log:
                print('START',stage,flush=True)
                result=subprocess.run([sys.executable,'-u','-m',module,*args],stdout=log,stderr=log)
            record=dict(stage=stage,exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
            with (ROOT/'campaign_stages.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            print(record,flush=True)
            if result.returncode:
                return result.returncode
        return 0
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    sys.exit(main())
