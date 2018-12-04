import subprocess


path='/home/saeid/software/sarsop/src/pomdpsol'
timeout = 20
subprocess.check_output([path, 'program.pomdp', \
            '--timeout', str(timeout), '--output', 'program.policy'])
