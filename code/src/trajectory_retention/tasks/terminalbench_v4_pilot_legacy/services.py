"""Capture unchanged native GUI services, including process memory, on private loopback.

Each run owns an anchor with network=none. Its worker and native services join only
that network namespace, using upstream's existing loopback port translation. No
host network, published ports, debugger, browser instrumentation or image changes.
Snapshots contain CRIU memory plus complete OCI filesystem deltas from frozen base
images. Both services stay paused through paired W/R capture. Private service bytes
are opaque to worker/review contexts; only snapshot hashes and lengths are listed.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from trajectory_retention.execution.environment import DockerWorkerEnvironment, CaptureError, RestoreError

RULE = 'legacy-native-process-runtime/v1'
MARK = '__task_services__/'
SERVICES = ('legacy-app', 'legacy-workstation')
PROFILE = 'tr-legacy'
HOSTS = ('legacy-app', 'legacy-workstation', 'main')
OWNER_LABEL = 'trajectory-retention.owner'
DIGEST = re.compile(r'^sha256:[0-9a-f]{64}$')
ID = re.compile(r'^[0-9a-f]{64}$')


def run(args, *, data=None, timeout=180, check=True):
    p = subprocess.run(args, input=data, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f'{args[0]} {args[1]} failed: {p.stderr.decode(errors="replace")[-3000:]}')
    return p


def docker(*args, **kw):
    return run(['docker', *args], **kw)


def guest(*args, **kw):
    return run(['colima', '--profile', PROFILE, 'ssh', '--', 'sudo', *args], **kw)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def runtime_environment():
    info=json.loads(docker('info','--format','{{json .}}').stdout)
    binaries={}
    for path in ('/usr/local/sbin/criu','/usr/bin/runc','/etc/criu/runc.conf'):
        binaries[path]=guest('sha256sum',path).stdout.decode().split()[0]
    return {'rule':RULE,'profile':PROFILE,'docker_name':info['Name'],
            'docker_version':info['ServerVersion'],'docker_root':info['DockerRootDir'],
            'kernel':info['KernelVersion'],'architecture':info['Architecture'],
            'experimental':info['ExperimentalBuild'],'binary_hashes':binaries,
            'criu_version':guest('criu','--version').stdout.decode().strip()}


def bundle(parts):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as tf:
        for name, data in parts.items():
            item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o600
            tf.addfile(item, io.BytesIO(data))
    return out.getvalue()


def members(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as tf:
        rows = tf.getmembers()
        if any(not m.isfile() for m in rows) or len({m.name for m in rows}) != len(rows):
            raise RestoreError('Unexpected or duplicate snapshot archive members')
        return {m.name: tf.extractfile(m).read() for m in rows}


def unbundle(data):
    parts = members(data)
    expected = {'worker.json', 'worker.tar', 'portal.tar', 'services.json'} | {
        name + suffix for name in SERVICES for suffix in ('.criu.tar.gz', '.oci.tar')}
    if set(parts) != expected:
        raise RestoreError('Incomplete native service runtime archive')
    binding = json.loads(parts['services.json'])
    if binding.get('rule') != RULE:
        raise RestoreError('Unexpected native service rule')
    for name, digest in binding['part_hashes'].items():
        if name not in parts or sha(parts[name]) != digest:
            raise RestoreError('Native service snapshot hash mismatch: ' + name)
    if set(binding['part_hashes']) != expected - {'worker.json', 'worker.tar', 'services.json'}:
        raise RestoreError('Native service snapshot hash coverage incomplete')
    return parts


def content(digest):
    if not DIGEST.fullmatch(digest):
        raise ValueError('Invalid OCI digest')
    data = guest('ctr', '-n', 'moby', 'content', 'get', digest).stdout
    if 'sha256:' + sha(data) != digest:
        raise CaptureError('OCI content hash mismatch')
    return data


def native_manifest(image):
    raw = content(image); doc = json.loads(raw)
    if 'manifests' in doc:
        matches = [m for m in doc['manifests'] if m.get('platform') == {'architecture':'arm64','os':'linux'}]
        if len(matches) != 1:
            raise CaptureError('Native image must have one linux/arm64 manifest')
        doc = json.loads(content(matches[0]['digest']))
    return doc


def export_delta(image, base):
    """OCI archive with only new blobs; native base layers must already be present."""
    raw = content(image); doc = json.loads(raw); base_doc = native_manifest(base)
    n = len(base_doc['layers'])
    if doc['layers'][:n] != base_doc['layers']:
        raise CaptureError('Service filesystem no longer extends its frozen native image')
    blobs = {image:raw, doc['config']['digest']:content(doc['config']['digest'])}
    for layer in doc['layers'][n:]:
        blobs[layer['digest']] = content(layer['digest'])
    index = {'schemaVersion':2, 'mediaType':'application/vnd.oci.image.index.v1+json', 'manifests':[
        {'mediaType':doc['mediaType'], 'digest':image, 'size':len(raw),
         'annotations':{'org.opencontainers.image.ref.name':'tr-legacy-snapshot:'+image.split(':')[1]}}]}
    return bundle({'oci-layout':b'{"imageLayoutVersion":"1.0.0"}', 'index.json':json.dumps(index).encode(),
                   **{'blobs/sha256/'+k.split(':')[1]:v for k,v in blobs.items()}})


def validate_delta(data, image, base):
    parts = members(data)
    for key, value in parts.items():
        if key.startswith('blobs/sha256/') and key != 'blobs/sha256/'+sha(value):
            raise RestoreError('OCI delta content hash mismatch')
    index = json.loads(parts['index.json'])
    if len(index['manifests']) != 1 or index['manifests'][0]['digest'] != image:
        raise RestoreError('OCI delta image binding differs')
    doc = json.loads(parts['blobs/sha256/'+image.split(':')[1]])
    base_doc = native_manifest(base); n = len(base_doc['layers'])
    if doc['layers'][:n] != base_doc['layers']:
        raise RestoreError('OCI delta does not extend the frozen native image')
    for row in [doc['config'], *doc['layers'][n:]]:
        value = parts.get('blobs/sha256/'+row['digest'].split(':')[1])
        if value is None or len(value) != row['size'] or 'sha256:'+sha(value) != row['digest']:
            raise RestoreError('OCI delta missing required content')


class ServiceWorkerEnvironment:
    def __init__(self, *, service_plan, image, timeout=300, network='none', **options):
        if network != 'none' or set(service_plan['services']) != set(SERVICES):
            raise ValueError('Legacy requires its two native services on private loopback')
        info = json.loads(docker('info','--format','{{json .}}').stdout)
        if info['Name'] != 'colima-tr-legacy' or info['ExperimentalBuild'] is not True:
            raise ValueError('Legacy requires the isolated colima-tr-legacy Docker context')
        self.runtime_environment=runtime_environment()
        if service_plan.get('runtime_environment') != self.runtime_environment:
            raise ValueError('Legacy process runtime differs from the frozen service plan')
        self.plan = service_plan
        self.owner = uuid.uuid4().hex
        self.anchor = None
        self.containers = {}
        self.paused = []
        self.worker = None
        self.snapshot_images = set()
        base = Path.home()/'.cache/trajectory-retention/legacy-services'; base.mkdir(parents=True,exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=self.owner[:10]+'-',dir=base))
        try:
            args = ['run','-d','--name','tr-legacy-net-'+self.owner[:12], '--network','none',
                    '--label',OWNER_LABEL+'='+self.owner, '--read-only', '--cap-drop','ALL']
            for host in HOSTS: args += ['--add-host',host+':127.0.0.1']
            self.anchor = docker(*args,image,'sh','-c','sleep infinity').stdout.decode().strip()
            for name in SERVICES:
                self.containers[name] = self._create_service(name)
                self._healthy(name)
            native_env = dict(v.split('=',1) for v in self.plan['main_config'].get('environment',[]))
            native_env.update(options.pop('env',{}) or {})
            self.worker = DockerWorkerEnvironment(image=image,timeout=timeout,network='container:'+self.anchor,
                                                   env=native_env,**options)
        except BaseException:
            self.close(); raise

    def __getattr__(self,name):
        worker = object.__getattribute__(self,'worker')
        if worker is None: raise AttributeError(name)
        return getattr(worker,name)

    def _create_service(self,name,image=None,start=True):
        svc = self.plan['services'][name]
        args = ['run' if start else 'create'] + (['-d'] if start else []) + [
            '--name','tr-legacy-'+self.owner[:10]+'-'+name+'-'+uuid.uuid4().hex[:6],
            '--label',OWNER_LABEL+'='+self.owner, '--network','container:'+self.anchor]
        for value in svc['config'].get('environment',[]): args += ['-e',value]
        if svc['config'].get('shm_size'): args += ['--shm-size',svc['config']['shm_size']]
        return docker(*args,image or svc['image_id']).stdout.decode().strip()

    def _healthy(self,name):
        test = self.plan['services'][name]['config']['healthcheck']['test']
        if test[0] != 'CMD-SHELL': raise ValueError('Unexpected native healthcheck')
        command = test[1].replace('$$','$'); deadline=time.monotonic()+120
        while True:
            p = docker('exec',self.containers[name],'sh','-c',command,check=False,timeout=10)
            if p.returncode == 0: return
            if time.monotonic()>deadline: raise RuntimeError('Native service health failed: '+name)
            time.sleep(.25)

    def binding(self):
        return {**self.worker.binding(),'network':'per-run-private-loopback/v1','runtime_rule':RULE,
                'task_services':{'rule':RULE,'owner':self.owner,'anchor':self.anchor,
                    'anchor_network':'none','containers':dict(self.containers),
                    'images':{n:self.plan['services'][n]['image_id'] for n in SERVICES},
                    'runtime_environment':self.runtime_environment}}

    def _active_connections(self):
        script = ("from pathlib import Path; import json; print(json.dumps([line for fn in "
                  "('/proc/net/tcp','/proc/net/tcp6') for line in Path(fn).read_text().splitlines()[1:] "
                  "if line.split()[3] in {'01','02','03','04','05','08','09','0B'}]))")
        return json.loads(docker('exec',self.anchor,'python3','-c',script).stdout)

    def _pause(self):
        if self.paused: return
        deadline=time.monotonic()+10
        while True:
            try:
                if not self._active_connections():
                    for cid in self.containers.values():
                        docker('pause',cid); self.paused.append(cid)
                    if not self._active_connections(): return
                self._unpause()
                if time.monotonic()>deadline:
                    raise CaptureError('Native services did not reach a connection-free command boundary')
                time.sleep(.1)
            except BaseException:
                self._unpause(); raise

    def _unpause(self):
        errors=[]
        for cid in self.paused:
            p=docker('unpause',cid,check=False)
            if p.returncode: errors.append(p.stderr.decode(errors='replace'))
        self.paused=[]
        if errors: raise CaptureError('Native services could not resume: '+str(errors))

    def _service_snapshot(self):
        if len(self.paused) != len(SERVICES): raise CaptureError('Services must be frozen together')
        parts={}; records={}
        for name in SERVICES:
            cid=self.containers[name]
            if not ID.fullmatch(cid): raise CaptureError('Invalid owned container ID')
            checkpoint='tr-'+uuid.uuid4().hex
            try:
                docker('checkpoint','create','--leave-running',cid,checkpoint)
                path=f'/var/lib/docker/containers/{cid}/checkpoints/{checkpoint}'
                parts[name+'.criu.tar.gz']=guest('tar','-C',path,'-czf','-','.').stdout
                committed=docker('commit','--no-pause',cid).stdout.decode().strip().splitlines()[-1]
                if not DIGEST.fullmatch(committed): raise CaptureError('Invalid committed image ID')
                self.snapshot_images.add(committed)
                parts[name+'.oci.tar']=export_delta(committed,self.plan['services'][name]['image_id'])
                records[name]={'snapshot_image_id':committed,'native_image_id':self.plan['services'][name]['image_id']}
            finally:
                docker('checkpoint','rm',cid,checkpoint,check=False)
        parts['portal.tar']=docker('cp',self.containers['legacy-app']+':/audit/.','-').stdout
        binding={'rule':RULE,'images':{n:self.plan['services'][n]['image_id'] for n in SERVICES},
                 'services':records,'portal_sha256':sha(parts['portal.tar']),
                 'part_hashes':{k:sha(v) for k,v in parts.items()},
                 'network':'per-run-private-loopback/v1','boundary':'all services paused; no active TCP connections',
                 'runtime_environment':self.runtime_environment}
        parts['services.json']=json.dumps(binding,sort_keys=True).encode()
        entries={MARK+n+'/native-state':{'type':'f','mode':0o600,'uid':0,'gid':0,
            'size':len(parts[n+'.criu.tar.gz'])+len(parts[n+'.oci.tar']),
            'sha256':sha(parts[n+'.criu.tar.gz']+parts[n+'.oci.tar'])} for n in SERVICES}
        return parts,binding,entries

    def runtime_manifest(self):
        self._pause()
        try:
            worker=self.worker.runtime_manifest(); _,_,entries=self._service_snapshot()
            return {**worker,'entries':{**worker['entries'],**entries}}
        finally:self._unpause()

    def capture_workspace(self):
        self._pause()
        try:return self.worker.capture_workspace()
        except BaseException:self._unpause(); raise

    def capture_runtime(self):
        self._pause()
        try:
            worker=self.worker.capture_runtime(); parts,binding,entries=self._service_snapshot()
            parts.update({'worker.json':json.dumps({k:v for k,v in worker.items() if k!='archive'}).encode(),
                          'worker.tar':worker.get('archive') or b''})
            result={**worker,'rule':RULE,'entries':{**worker['entries'],**entries},
                    'archive':bundle(parts),'coverage':{**worker['coverage'],
                    'task_services':binding,'native_service_processes':'full CRIU checkpoint plus OCI filesystem delta'}}
            result['changed']=sorted(result['entries'])
            return result
        finally:self._unpause()

    def restore_runtime(self,runtime):
        if runtime.get('rule') != RULE or not runtime.get('archive'):
            raise RestoreError('Missing complete native service runtime')
        parts=unbundle(runtime['archive']); binding=json.loads(parts['services.json'])
        if binding['images'] != {n:self.plan['services'][n]['image_id'] for n in SERVICES}:
            raise RestoreError('Native service image binding differs')
        if binding.get('runtime_environment') != self.runtime_environment:
            raise RestoreError('Native process runtime differs from the captured environment')
        worker=json.loads(parts['worker.json']); worker['archive']=parts['worker.tar'] or None
        result=self.worker.restore_runtime(worker)
        for name,cid in list(self.containers.items()):
            docker('rm','-f',cid); del self.containers[name]
        for name in SERVICES:
            image=binding['services'][name]['snapshot_image_id']
            validate_delta(parts[name+'.oci.tar'],image,self.plan['services'][name]['image_id'])
            docker('load',data=parts[name+'.oci.tar']); self.snapshot_images.add(image)
            cid=self._create_service(name,image=image,start=False); self.containers[name]=cid
            checkpoint='tr-'+uuid.uuid4().hex
            path=f'/var/lib/docker/containers/{cid}/checkpoints/{checkpoint}'
            guest('mkdir','-p',path)
            # CRIU images are immutable harness data, never supplied by a worker.
            guest('tar','-C',path,'-xzf','-',data=parts[name+'.criu.tar.gz'])
            docker('start','--checkpoint',checkpoint,cid)
            docker('checkpoint','rm',cid,checkpoint)
            self._healthy(name)
        return {**result,'rule':RULE,'services_restored':True,'native_service_processes_restored':True,
                'service_binding':binding,'fresh_private_loopback':True}

    def close(self):
        errors=[]
        if self.paused:
            try:self._unpause()
            except Exception as exc:errors.append(str(exc))
        if self.worker is not None:
            try:self.worker.close()
            except Exception as exc:errors.append(str(exc))
        for cid in [*self.containers.values(),*([self.anchor] if self.anchor else [])]:
            p=docker('rm','-f',cid,check=False)
            if p.returncode:errors.append(p.stderr.decode(errors='replace'))
        self.containers={};self.anchor=None
        for image in self.snapshot_images:
            # Never force removal: another live branch may use the same immutable image.
            docker('image','rm','--no-prune',image,check=False)
        if not errors:shutil.rmtree(self.root,ignore_errors=True)
        if errors:raise RuntimeError('Owned legacy service cleanup failed: '+str(errors))
