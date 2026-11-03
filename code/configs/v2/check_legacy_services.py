"""Native legacy network/visibility admission checks, without provider calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
from trajectory_retention.tasks.terminalbench_v4_pilot_legacy.services import ServiceWorkerEnvironment, docker, guest

PROBE = r'''
import json,socket,urllib.request,urllib.error
from pathlib import Path
def connect(host,port):
    try:
        with socket.create_connection((host,port),timeout=.5):return True
    except OSError:return False
def status(path):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18077'+path,timeout=2) as r:return r.status
    except urllib.error.HTTPError as e:return e.code
print(json.dumps({'interfaces':sorted(p.name for p in Path('/sys/class/net').iterdir()),
    'routes':Path('/proc/net/route').read_text().splitlines()[1:],
    'external_ip_connects':connect('1.1.1.1',443),
    'host_gateway_connects':connect('192.168.5.2',22),
    'app_health':status('/health'),'private_cases_without_session':status('/api/cases'),
    'page_without_workstation_token':status('/'),
    'vnc_connects':connect('127.0.0.1',18078),'debugger_connects':connect('127.0.0.1',9222)}))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('artifacts/v2/lane-v4-pilot-legacy/services-check.json'))
    args=ap.parse_args();plan=json.loads(Path('artifacts/v2/lane-v4-pilot-legacy/service-runtime-plan.json').read_text())
    build=json.loads(Path('artifacts/v2/lane-v4-pilot-legacy/tasks/legacy-utility-triage/worker-build.json').read_text())
    receipt={'rule':'legacy-service-isolation-check/v1','provider_calls':0,'runs':[],'assertions':{}}
    environments=[]
    try:
        for _ in range(2):
            env=ServiceWorkerEnvironment(service_plan=plan,image=build['image']['id'],memory='8192m',cpus='2')
            environments.append(env)
            result=json.loads(env._exec('python3 -c '+shlex.quote(PROBE)).stdout)
            ids=[env.anchor,env.container_id,*env.containers.values()]
            namespaces=[]
            for cid in ids:
                info=json.loads(docker('inspect',cid).stdout)[0]
                assert not info['HostConfig']['Privileged'] and not info['HostConfig']['PortBindings']
                namespaces.append(guest('readlink',f"/proc/{info['State']['Pid']}/ns/net").stdout.decode().strip())
            receipt['runs'].append({'binding':env.binding(),'probe':result,'network_namespaces':namespaces})
        a=receipt['assertions']
        a['each_run_shares_only_its_own_namespace']=all(len(set(r['network_namespaces']))==1 for r in receipt['runs'])
        a['runs_have_different_network_namespaces']=receipt['runs'][0]['network_namespaces'][0]!=receipt['runs'][1]['network_namespaces'][0]
        a['loopback_is_the_only_interface']=all(r['probe']['interfaces']==['lo'] for r in receipt['runs'])
        a['no_external_routes']=all(r['probe']['routes']==[] for r in receipt['runs'])
        a['external_and_host_connections_blocked']=all(not r['probe']['external_ip_connects'] and not r['probe']['host_gateway_connects'] for r in receipt['runs'])
        a['native_services_reachable']=all(r['probe']['app_health']==200 and r['probe']['vnc_connects'] for r in receipt['runs'])
        a['native_gui_access_controls_preserved']=all(r['probe']['private_cases_without_session']==403 and r['probe']['page_without_workstation_token']==403 and not r['probe']['debugger_connects'] for r in receipt['runs'])
        a['no_published_ports_or_privileged_containers']=True
        receipt['passed']=all(a.values())
    except Exception as exc:
        receipt['passed']=False;receipt['error']=f'{type(exc).__name__}: {exc}'
    finally:
        for env in environments:
            try:env.close()
            except Exception as exc:receipt['passed']=False;receipt.setdefault('cleanup_errors',[]).append(str(exc))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':receipt['passed'],'assertions':receipt['assertions'],'error':receipt.get('error')},indent=2))
    raise SystemExit(0 if receipt['passed'] else 1)

if __name__=='__main__':main()
