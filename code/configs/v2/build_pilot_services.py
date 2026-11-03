"""Build newly authorized service tasks natively, retaining image and base receipts; no model calls."""
from pathlib import Path
import argparse
import importlib
import json
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
TASKS = {'heat-pump-warranty': 'heatpump', 'legacy-utility-triage': 'legacy'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=TASKS)
    args = ap.parse_args()
    module = 'trajectory_retention.tasks.terminalbench_v4_pilot_' + TASKS[args.task]
    E = importlib.import_module(module + '.environment')
    O = importlib.import_module(module + '.official')
    S = importlib.import_module(module + '.source')
    P = importlib.import_module(module + '.package')
    S.verify_checkout()
    built = P.build_images(args.task)
    if built['status'] != 'BUILT':
        raise RuntimeError(json.dumps(built))
    print(json.dumps({'task': args.task, 'main_images': 'built'}), flush=True)
    source = S.task_dir(args.task)
    import yaml
    compose = yaml.safe_load((source / 'environment/docker-compose.yaml').read_text())
    manifest_rows = next(r for r in O.release_manifest()['tasks'] if r['task'] == args.task)
    services = {}
    for name, config in compose['services'].items():
        if name == 'main':
            continue
        role = 'sidecar-' + name
        row = next(x for x in manifest_rows['images'] if x['role'] == role)
        context = source / row['context']
        dockerfile = source / row['dockerfile']
        work = E.BUILD_ROOT / args.task / role
        receipt = S.EVIDENCE / 'tasks' / args.task / (role + '-build.json')
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior.get('result') == 'built':
                try:
                    assert E.image_info(prior['image']['id'])['architecture'] == 'arm64'
                    services[name] = {'image_id': prior['image']['id'], 'config': config}
                    continue
                except Exception:
                    pass
        official, _ = O.official_config(args.task, role)
        bases = []
        def pin(match):
            ref = match.group(2)
            # Non-Docker-Hub bases keep their explicit upstream version, resolving its native digest.
            if ref.startswith('mcr.microsoft.com/'):
                found = E.resolve_base(ref)
                found['match'] = 'upstream version tag, native repository digest'
            else:
                found = O.match_base(ref, official)
                if found.get('pinned'):
                    O.pull_arm64(found['pinned'])
                else:
                    found.update(E.resolve_base(ref))
                    found['match'] = 'fallback current upstream tag; disclosed'
            bases.append(found)
            return match.group(1) + found['pinned'] + match.group(3)
        text = re.sub(r'(?m)^(FROM\s+)(\S+)(.*)$', pin, dockerfile.read_text())
        result = E._build(context, E.TAG.format(task=args.task, kind=role), dockerfile_text=text,
                          workdir=work, timeout=1800,
                          labels={'terminalbench.task': args.task, 'terminalbench.role': role})
        result.update(bases=bases, upstream_dockerfile_sha256=S.sha256_file(dockerfile),
                      context_manifest=E.context_manifest(context), official_image=row['pinned_ref'])
        P.write_json(receipt, result)
        if result['result'] != 'built':
            raise RuntimeError(json.dumps({'task': args.task, 'role': role, 'result': result}))
        assert result['image']['architecture'] == 'arm64'
        services[name] = {'image_id': result['image']['id'], 'config': config}
        print(json.dumps({'task': args.task, 'service': name, 'status': 'built'}), flush=True)
    plan = {'task': args.task, 'lane': S.LANE, 'source_revision': S.SOURCE_REVISION,
            'services': services, 'main_config': compose['services']['main'],
            'artifacts': tomllib.loads((source / 'task.toml').read_text())['artifacts'],
            'network_policy': 'per-run internal bridge; isolated gateway; no host ports; no external network',
            'admitted': False, 'provider_calls': 0}
    P.write_json(S.EVIDENCE / 'service-build.json', plan)
    print(json.dumps({'task': args.task, 'builds_complete': True, 'admitted': False}), flush=True)

if __name__ == '__main__':
    main()
