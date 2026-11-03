"""Per-run offline task services; immutable servers plus captured portal audit state.

Only the portal writes state, under /audit. Its handler reloads that state per request.
All other service data is immutable and bound to native image IDs. Service root filesystems
are read-only. No host ports, host gateway or external routes are made available.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from trajectory_retention.execution.environment import DockerWorkerEnvironment, CaptureError, RestoreError, fidelity_manifest_from_tar
from trajectory_retention.execution.archives import extract_directory

RULE = 'heatpump-services-runtime/v1'
MARK = '__task_services__/'

def docker(*args, data=None, timeout=120, check=True):
    p = subprocess.run(['docker', *args], input=data, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f'Docker operation {args[0]} failed: {p.stderr.decode(errors="replace")[-1200:]}')
    return p

def bundle(parts):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as tf:
        for name, data in parts.items():
            item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o600
            tf.addfile(item, io.BytesIO(data))
    return out.getvalue()

def unbundle(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        files = {m.name: tf.extractfile(m).read() for m in tf if m.isfile()}
    if set(files) != {'worker.json', 'worker.tar', 'portal.tar', 'services.json'}:
        raise RestoreError('Unexpected service runtime archive members')
    return files

class ServiceWorkerEnvironment:
    def __init__(self, *, service_plan, image, timeout=300, network='none', **options):
        if network != 'none':
            raise ValueError('Service tasks must not inherit an external network')
        self.plan = service_plan
        self.owner = uuid.uuid4().hex
        self.network_id = None
        self.containers = {}
        self.paused = []
        self.worker = None
        base = Path.home() / '.cache/trajectory-retention/heatpump-services'
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=self.owner[:8]+'-', dir=base))
        self.audit = self.root / 'audit'; self.audit.mkdir()
        try:
            self.network_id = docker('network','create','--internal','--driver','bridge',
                '--opt','com.docker.network.bridge.gateway_mode_ipv4=isolated',
                '--label','trajectory-retention.owner='+self.owner, 'tr-heatpump-'+self.owner[:12]).stdout.decode().strip()
            for name, service in self.plan['services'].items():
                args = ['run','-d','--name','tr-heatpump-'+self.owner[:10]+'-'+name,
                    '--label','trajectory-retention.owner='+self.owner,
                    '--network',self.network_id,'--network-alias',name,'--read-only',
                    '-e','PYTHONDONTWRITEBYTECODE=1']
                if name == 'warranty-portal':
                    args += ['--mount',f'type=bind,source={self.audit},target=/audit']
                args += [service['image_id']]
                self.containers[name] = docker(*args).stdout.decode().strip()
            for name, cid in self.containers.items():
                health = self.plan['services'][name]['config']['healthcheck']['test']
                assert health[0] == 'CMD'
                deadline = time.monotonic()+120
                while True:
                    p = docker('exec',cid,*health[1:],check=False,timeout=10)
                    if p.returncode == 0: break
                    if time.monotonic() > deadline: raise RuntimeError('Service did not become healthy: '+name)
                    time.sleep(.25)
            native_env = dict(v.split('=', 1) for v in self.plan['main_config'].get('environment', []))
            native_env.update(options.pop('env', {}) or {})
            self.worker = DockerWorkerEnvironment(image=image,timeout=timeout,network=self.network_id,env=native_env,**options)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        worker = object.__getattribute__(self,'worker')
        if worker is None: raise AttributeError(name)
        return getattr(worker,name)

    def binding(self):
        return {**self.worker.binding(),'network':'per-run-internal-isolated/v1','runtime_rule':RULE,'task_services':{'rule':RULE,'owner':self.owner,
            'network':self.network_id,'internal':True,'gateway_mode':'isolated',
            'containers':dict(self.containers),'images':{k:v['image_id'] for k,v in self.plan['services'].items()}}}

    def _pause(self):
        if self.paused: return
        try:
            for cid in self.containers.values():
                probe = docker('exec',cid,'python3','-c',
                    "from pathlib import Path; import json; "
                    "active=[line for fn in ('/proc/net/tcp','/proc/net/tcp6') "
                    "for line in Path(fn).read_text().splitlines()[1:] "
                    "if line.split()[3] in {'01','02','03','04','05','08','09','0B'}]; print(json.dumps(active))")
                if json.loads(probe.stdout):
                    raise CaptureError('Task service has an active connection at capture')
                docker('pause',cid); self.paused.append(cid)
        except BaseException:
            self._unpause(); raise

    def _unpause(self):
        failed = []
        for cid in self.paused:
            p = docker('unpause',cid,check=False)
            if p.returncode: failed.append(p.stderr.decode())
        self.paused = []
        if failed: raise CaptureError('Could not unpause owned services: '+str(failed))

    def _portal(self):
        return docker('cp',self.containers['warranty-portal']+':/audit/.','-').stdout

    def _entries(self, archive):
        return {MARK+'warranty-portal/audit/'+k:v for k,v in fidelity_manifest_from_tar(archive).items()}

    def runtime_manifest(self):
        own_pause = not self.paused
        if own_pause: self._pause()
        try:
            w = self.worker.runtime_manifest()
            return {**w,'entries':{**w['entries'],**self._entries(self._portal())}}
        finally:
            if own_pause: self._unpause()

    def capture_workspace(self):
        # Keep services frozen through the paired W/R capture. At a valid boundary no
        # worker command or HTTP handler may still be in flight (checked below).
        self._pause()
        try:
            return self.worker.capture_workspace()
        except BaseException:
            self._unpause(); raise

    def capture_runtime(self):
        self._pause()
        try:
            w = self.worker.capture_runtime()
            portal = self._portal()
            service_binding = {'rule':RULE,'images':{k:v['image_id'] for k,v in self.plan['services'].items()},
                'mutable_state':'warranty-portal:/audit','root_filesystems':'read-only',
                'portal_sha256':hashlib.sha256(portal).hexdigest()}
            state = {**w,'entries':{**w['entries'],**self._entries(portal)},'rule':RULE,
                'coverage':{**w['coverage'],'task_services':service_binding},
                'archive':bundle({'worker.json':json.dumps({k:v for k,v in w.items() if k!='archive'}).encode(),
                                  'worker.tar':w.get('archive') or b'', 'portal.tar':portal,
                                  'services.json':json.dumps(service_binding).encode()})}
            state['changed'] = sorted(state['entries'])
            return state
        finally:
            self._unpause()

    def restore_runtime(self, runtime):
        if runtime.get('rule') != RULE or not runtime.get('archive'):
            raise RestoreError('Missing complete service runtime archive')
        parts = unbundle(runtime['archive']); binding = json.loads(parts['services.json'])
        expected = {k:v['image_id'] for k,v in self.plan['services'].items()}
        if binding['images'] != expected or hashlib.sha256(parts['portal.tar']).hexdigest()!=binding['portal_sha256']:
            raise RestoreError('Service snapshot image/hash mismatch')
        w = json.loads(parts['worker.json']); w['archive'] = parts['worker.tar'] or None
        result = self.worker.restore_runtime(w)
        self._pause()
        try:
            for path in list(self.audit.iterdir()):
                if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
                else: path.unlink()
            # Docker's archive copy retains numeric ownership/modes inside the bound volume.
            docker('cp','-',self.containers['warranty-portal']+':/audit',data=parts['portal.tar'])
            live = self._entries(self._portal())
            from trajectory_retention.execution.environment import comparable
            if comparable(live) != comparable(self._entries(parts['portal.tar'])):
                raise RestoreError('Portal audit restore differs from capture')
        finally:
            self._unpause()
        return {**result,'rule':RULE,'services_restored':True,'service_binding':binding}

    def close(self):
        errors = []
        if self.paused:
            try: self._unpause()
            except Exception as e: errors.append(str(e))
        if self.worker is not None:
            try: self.worker.close()
            except Exception as e: errors.append(str(e))
        for cid in self.containers.values():
            p = docker('rm','-f',cid,check=False)
            if p.returncode: errors.append(p.stderr.decode())
        self.containers = {}
        if self.network_id:
            p = docker('network','rm',self.network_id,check=False)
            if p.returncode: errors.append(p.stderr.decode())
            self.network_id = None
        # Captured state is owned scratch; leave it on cleanup failures for diagnosis.
        if not errors: shutil.rmtree(self.root,ignore_errors=True)
        if errors: raise RuntimeError('Owned service cleanup failed: '+str(errors))
