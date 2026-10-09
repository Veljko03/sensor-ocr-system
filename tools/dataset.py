"""Import historical originals without changing them; validate/export manifests."""
import argparse
import hashlib
import json
from pathlib import Path
from .dataset_store import Store, import_legacy, normalize_value


def validate(store):
    rows=store.rows()
    hashes=set(); groups={}; failures=[]
    for row in rows:
        path=store.image_path(row['id'])
        if row['sha256'] in hashes:
            failures.append('Duplicate content hash: '+row['id'])
        hashes.add(row['sha256'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            failures.append('Image hash changed: '+row['id'])
        if row['group_id'] in groups and groups[row['group_id']]!=row['split']:
            failures.append('Group appears in multiple partitions: '+row['group_id'])
        groups[row['group_id']]=row['split']
        if row['class']=='readable':
            try:
                normalize_value(row['value'])
            except ValueError:
                failures.append('Invalid ground truth: '+row['id'])
        elif row['value'] is not None:
            failures.append('Non-readable image has numeric value: '+row['id'])
    return {'valid':not failures,'errors':failures,'version':store.version(),
            'summary':store.summary(),'groups':groups}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import-legacy');imp.add_argument('--project',required=True)
    sub.add_parser('stats');sub.add_parser('validate')
    export=sub.add_parser('export');export.add_argument('--output',required=True)
    args=parser.parse_args();store=Store(args.dataset)
    if args.command=='import-legacy':
        result=import_legacy(store,args.project)
    elif args.command=='validate':
        result=validate(store)
    elif args.command=='export':
        path=Path(args.output)
        with path.open('x',encoding='utf-8',newline='') as file:
            file.write(store.export_csv())
        result={'output':str(path),'version':store.version()}
    else:
        result=store.summary()
    print(json.dumps(result,indent=2))
    if result.get('valid') is False:
        raise SystemExit(1)


if __name__=='__main__':
    main()
