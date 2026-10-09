"""Reproducible frame-level evaluation; final test data requires explicit opt-in."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import statistics
import time


def summarize(rows):
    positive=[r for r in rows if r['class']=='readable']
    absent=[r for r in rows if r['class']=='absent']
    unreadable=[r for r in rows if r['class']=='unreadable']
    ambiguous=[r for r in rows if r['class']=='unreadable_or_absent']
    negative=absent+unreadable+ambiguous
    accepted=[r for r in rows if r['prediction'] is not None]
    wrong=[r for r in accepted if r['class']!='readable' or r['prediction']!=r['value']]
    correct=[r for r in positive if r['prediction']==r['value']]
    digit_correct=0
    for row in positive:
        if row['prediction'] is not None:
            actual=row['prediction'].replace('.','')
            expected=row['value'].replace('.','')
            digit_correct+=sum(a==b for a,b in zip(actual,expected))
    def ratio(n,d):
        return n/d if d else None
    return {'readable':len(positive),'unreadable':len(unreadable),'no_display':len(absent),
        'unreadable_or_absent_unspecified':len(ambiguous),
        'correctly_accepted':len(correct),'incorrectly_accepted':len(wrong),
        'rejected_readable':sum(r['prediction'] is None for r in positive),
        'correct_acceptance_rate':ratio(len(correct),len(positive)),
        'false_acceptance_rate_negative':ratio(sum(r['prediction'] is not None for r in negative),len(negative)),
        'false_acceptance_rate_no_display':ratio(sum(r['prediction'] is not None for r in absent),len(absent)),
        'false_acceptance_rate_unreadable':ratio(sum(r['prediction'] is not None for r in unreadable),len(unreadable)),
        'numerical_error_rate_among_accepted':ratio(len(wrong),len(accepted)),
        'digit_accuracy_readable_frames_rejections_count_as_incorrect':ratio(digit_correct,3*len(positive)),
        'full_reading_exact_match_rate':ratio(len(correct),len(positive))}


def load_manifest(args):
    if args.legacy:
        root=Path(args.legacy).resolve()
        with (root/'labels.csv').open(newline='',encoding='utf-8') as file:
            raw=list(csv.DictReader(file))
        rows=[]
        for row in raw:
            if Path(row['filename']).name!=row['filename']:
                raise ValueError('Invalid legacy filename')
            path=root/'images'/row['filename']
            rows.append({'filename':row['filename'],'path':str(path),'class':'readable' if row['readable']=='1' else ('absent' if row['note']=='no_meter' else 'unreadable'),
                'value':row['display_text'] or None,'group_id':'legacy_20260910','split':'legacy',
                'angle':'unknown','distance':'unknown','lighting':'unknown',
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return rows,hashlib.sha256((root/'labels.csv').read_bytes()).hexdigest()
    root=Path(args.manifest).resolve().parent
    with Path(args.manifest).open(newline='',encoding='utf-8') as file:
        rows=list(csv.DictReader(file))
    groups={};hashes={}
    for row in rows:
        path=(root/row['filename']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes manifest root')
        row['path']=str(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Image checksum mismatch: '+row['filename'])
        group=row['group_id']
        if group in groups and groups[group]!=row['split']:
            raise ValueError('Group leakage across partitions: '+group)
        groups[group]=row['split']
        if row['sha256'] in hashes:
            raise ValueError('Duplicate image content in manifest')
        hashes[row['sha256']]=row['split']
    return [r for r in rows if r['split']==args.split],hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--legacy',help='Path to dataset/real_meter')
    inputs.add_argument('--manifest',help='Export CSV next to its images directory')
    parser.add_argument('--split',choices=['legacy','development','validation','test'],default='development')
    parser.add_argument('--final-test',action='store_true')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--ocr',default=str(Path(__file__).resolve().parents[1]/'real_meter_ocr.py'))
    parser.add_argument('--confidence',type=float,default=.45)
    parser.add_argument('--output',required=True,help='New JSON result path; refuses overwrite')
    args=parser.parse_args()
    if not 0<=args.confidence<=1:
        parser.error('Confidence must be between 0 and 1')
    if not args.legacy and args.split=='test' and (not args.final_test or args.allow_incomplete):
        parser.error('Final test requires --final-test and fully completed human labels')
    rows,version=load_manifest(args)
    if not rows:
        parser.error('No images in selected split')
    pending=[r for r in rows if r['class'] not in {'readable','unreadable','absent'}]
    if pending and not args.allow_incomplete:
        parser.error(str(len(pending))+' images are pending/skipped; finish human labeling')
    from .dataset_store import normalize_value
    for row in rows:
        if row['class']=='readable':
            row['value']=normalize_value(row['value'])
        elif row.get('value'):
            raise ValueError('Negative image has a numerical label')
        if row['class']=='unreadable' and row.get('label_origin')=='human_simple':
            row['class']='unreadable_or_absent'
    import cv2
    cv2.setNumThreads(1)
    spec=importlib.util.spec_from_file_location('candidate_ocr',args.ocr)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cpu_before=resource.getrusage(resource.RUSAGE_SELF)
    started=time.perf_counter();result=[]
    for row in rows:
        if row in pending:
            continue
        image=cv2.imread(row['path'])
        if image is None:
            raise ValueError('Image cannot be decoded: '+row['path'])
        before=time.perf_counter()
        value,debug=module.read_real_meter_image(image,confidence_threshold=args.confidence)
        elapsed=time.perf_counter()-before
        if value is not None and (not math.isfinite(value) or not 0<=value<=9.99):
            raise ValueError('OCR violated esp_1 numerical contract')
        prediction=f'{value:.2f}' if value is not None else None
        result.append(dict(row,prediction=prediction,latency_seconds=elapsed,debug=debug))
    timings=[r['latency_seconds'] for r in result]
    if not timings:
        parser.error('No reviewed images in selected split')
    after=resource.getrusage(resource.RUSAGE_SELF)
    report={'schema_version':1,'dataset_version':version,'ocr_sha256':hashlib.sha256(Path(args.ocr).read_bytes()).hexdigest(),
        'confidence':args.confidence,'split':'legacy' if args.legacy else args.split,
        'configuration':module.evaluation_configuration() if hasattr(module,'evaluation_configuration') else {},
        'level':'frame','designated_holdout_partition':bool(args.final_test and not args.legacy),
        'unreviewed_or_skipped':len(pending),'metrics':summarize(result),
        'latency_seconds':{'mean':statistics.mean(timings),'p95':sorted(timings)[int(.95*(len(timings)-1))],'max':max(timings)},
        'wall_seconds':time.perf_counter()-started,'cpu_seconds':after.ru_utime+after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
        'peak_process_rss_kib':after.ru_maxrss,'opencv_threads':cv2.getNumThreads(),
        'by_condition':{},'rows':result,
        'limitations':['Frame metrics do not establish temporal correctness.','Confidence is heuristic, not a calibrated probability.',
                      'Independence requires meaningful capture groups; a partition name alone is insufficient.',
                      'Unknown conditions do not establish angle/distance/lighting coverage.','A finite test does not guarantee zero future false positives.']}
    for field in ['angle','distance','lighting','group_id']:
        report['by_condition'][field]={key:summarize([r for r in result if r[field]==key]) for key in sorted(set(r[field] for r in result))}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(report,indent=2)
    with output.open('x',encoding='utf-8') as file:
        file.write(serialized)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','by_condition']},indent=2))


if __name__=='__main__':
    main()
