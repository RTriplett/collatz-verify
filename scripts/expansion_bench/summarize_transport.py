#!/usr/bin/env python3
"""Audit and summarize completed transport experiments; never pool unlike runs."""
import argparse
import json
from pathlib import Path
from summarize_overall import check_report, digest, distribution, require, work_signature


def summarize(path):
    metadata = json.loads((path / 'metadata.json').read_text())
    require(metadata.get('state') == 'complete' and metadata.get('exit_status') == 0, f'{path}: incomplete')
    rows = [json.loads(s) for s in (path / 'pipeline.jsonl').read_text().splitlines() if s.strip()]
    require(bool(rows), 'empty measurement log')
    expected_transport = {'replay16':'replay16','packed8':'packed8','affine16':'replay16',
                          'affine32':'affine32','parents':'parents'}
    for row in rows:
        if row['type'] in ('validation','warmup','trial'):
            require(row['variant'] in expected_transport, 'unknown variant')
            require(row['report']['gpu']['transport'] == expected_transport[row['variant']], 'variant/transport mismatch')
    if all(r['type'] == 'validation' for r in rows):
        signature = None
        for row in rows:
            require(row['cpu_gpu_equal'] and row['same_policy_equal'], 'validation failed')
            check_report(row['report'])
            sig = work_signature(row['report'])
            require(signature is None or signature == sig, 'validation work differs')
            signature = sig
        return {'directory':str(path),'profile':'validation','comparisons':len(rows),
                'records':rows,'runner':metadata}
    configs = [r for r in rows if r['type'] == 'config']
    require(len(configs) == 1, 'missing/duplicate config')
    cfg = configs[0]
    trials = [r for r in rows if r['type'] == 'trial']
    require(len(trials) == len(cfg['variants']) * cfg['repetitions'], 'incomplete trials')
    warmups = [r for r in rows if r['type'] == 'warmup']
    require(sorted(r['variant'] for r in warmups) == sorted(cfg['variants']), 'incomplete warmups')
    warm_signature = None
    for row in warmups:
        check_report(row['report'])
        sig = work_signature(row['report'])
        require(warm_signature is None or warm_signature == sig, 'warmup work differs')
        warm_signature = sig
    expected = None
    for row in trials:
        check_report(row['report'],cfg)
        sig = work_signature(row['report'])
        require(expected is None or expected == sig, 'same-work mismatch')
        expected = sig
    variants = []
    baseline = {r['repetition']:r for r in trials if r['variant']=='replay16'}
    for name in cfg['variants']:
        selected = [r for r in trials if r['variant']==name]
        require(sorted(r['repetition'] for r in selected)==list(range(cfg['repetitions'])), 'missing/duplicate repetition')
        g = selected[0]['report']['gpu']
        if g['transport'] != 'parents':
            require(all(r['report']['gpu']['transfer_bytes']==g['transfer_bytes'] for r in selected), 'transfer count differs')
        transfer = distribution(r['report']['gpu']['transfer_bytes'] for r in selected)
        records = distribution(r['report']['gpu']['input_records'] for r in selected)
        item = {'name':name,'trials':selected,
                **{k:distribution(r[k] for r in selected) for k in ('wall_seconds','process_cpu_seconds','average_busy_cpu_cores')},
                'transfer_bytes':transfer['median'],'input_records':records['median'],
                'transfer_bytes_distribution':transfer,'input_records_distribution':records,
                'transfer_gb_per_second':distribution(r['report']['gpu']['transfer_bytes']/r['wall_seconds']/1e9 for r in selected)}
        if baseline:
            item['paired_wall_ratio']=distribution(r['wall_seconds']/baseline[r['repetition']]['wall_seconds'] for r in selected)
            item['paired_cpu_ratio']=distribution(r['process_cpu_seconds']/baseline[r['repetition']]['process_cpu_seconds'] for r in selected)
        variants.append(item)
    setups = [r for r in rows if r['type']=='gpu_setup']
    require(len(setups)==1,'setup record missing')
    return {'directory':str(path),'profile':'target','config':cfg,'corpus_sha256':digest(cfg['roots']),
            'work_signature_sha256':digest(expected),'variants':variants,'setup':setups[0],
            'class_probe':[r for r in rows if r['type']=='class_probe'],'runner':metadata}


def interval(d):
    return f"{d['median']:.3f} [{d['minimum']:.3f}, {d['maximum']:.3f}]"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('runs',type=Path,nargs='+')
    args=p.parse_args()
    result={'scope':'Resident-array complete CPU/GPU pipeline; sampled N72 packages, not full 2^72 verification.',
            'notes':'Observed ranges are not confidence intervals. Paired ratios use repetition within one run. CUDA event intervals overlap. Parent traffic includes repeated group headers and batch splits.',
            'runs':[summarize(path) for path in args.runs]}
    lines=['# Transport and terminal reuse experiments','',result['scope'],'',result['notes'],'']
    for r in result['runs']:
        lines += [f"## {r['directory']}",'']
        if r['profile']=='validation':
            first=r['records'][0]
            lines += [f"{r['comparisons']} CPU/GPU comparisons passed; {first['scope']}, A={first['a']}.",'']
            continue
        cfg=r['config']
        lines += [f"{len(cfg['roots'])} K38 roots; {cfg['threads']} workers; A={cfg['a']}; cutoff50; table entry52; {cfg['repetitions']} rotating repetitions.",'',
                  '| Variant | Wall s, median [min, max] | CPU core-s, median [min, max] | Busy cores | Transfer GB | Paired wall ratio |',
                  '|---|---:|---:|---:|---:|---:|']
        for v in r['variants']:
            ratio=v.get('paired_wall_ratio',{}).get('median')
            lines.append(f"| {v['name']} | {interval(v['wall_seconds'])} | {interval(v['process_cpu_seconds'])} | {v['average_busy_cpu_cores']['median']:.3f} | {v['transfer_bytes']/1e9:.3f} | {ratio:.4f} |" if ratio is not None else f"| {v['name']} | {interval(v['wall_seconds'])} | {interval(v['process_cpu_seconds'])} | {v['average_busy_cpu_cores']['median']:.3f} | {v['transfer_bytes']/1e9:.3f} | — |")
        lines += ['',f"GPU setup {r['setup']['wall_seconds']:.3f}s, outside trials. Corpus `{r['corpus_sha256']}`; all work signatures match.",'']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    args.out.with_suffix('.md').write_text('\n'.join(lines))
    print(f'Saved {args.out}')


if __name__=='__main__':
    main()
