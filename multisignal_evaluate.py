"""Validate raw interventions, train two ablations and run protected evaluation."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
import zipfile
import numpy as np
from multisignal_study import (ROOT, SEED, read, sha, digest, write_new, source_manifest,
                              jobs_outputs, make_job, run_jobs, utc)


def pairs_and_audit():
    from mapf.multisignal_teacher import compare_evidence
    from mapf.grid import GridMap
    from mapf.validation import validate_solution
    manifest = read(ROOT/'manifest.json')
    grid = GridMap.from_file(manifest['map_path'])
    grouped = defaultdict(dict)
    statuses = Counter()
    errors = []
    for job, result in jobs_outputs(ROOT/'replays'):
        monitor_path = ROOT/'replays'/(job['job_id']+'.monitor.json')
        if not result or not monitor_path.exists():
            errors.append(dict(job_id=job['job_id'], reason='Missing result or monitor'))
            continue
        monitor = read(monitor_path)
        if result['job_sha256'] != digest(job) or monitor['job_sha256'] != digest(job):
            raise ValueError('Job identity corruption')
        if sha(ROOT/'replays'/(job['job_id']+'.json')) != monitor['output_sha256']:
            raise ValueError('Raw output checksum differs')
        if result['status'] != 'OK':
            errors.append(dict(job_id=job['job_id'], reason=result.get('traceback','ERROR')))
            continue
        evidence = result['data']
        statuses[evidence['status']] += 1
        if evidence['status'] == 'SOLVED':
            instance=job['instance']
            validate_solution(grid,list(map(tuple,instance['starts'])),list(map(tuple,instance['goals'])),
                              [list(map(tuple,p)) for p in evidence['paths']], evidence['cost'])
        for metric in ('nodes_expanded','ll_expansions','astar_calls'):
            if evidence.get('intervention_applied') and evidence['prefix_'+metric]+evidence['suffix_'+metric] != evidence[metric]:
                raise ValueError('Prefix/suffix accounting mismatch')
        group=grouped[job['query_id']]
        if job['candidate_index'] in group:
            raise ValueError('Duplicate query candidate')
        group[job['candidate_index']] = (job,evidence)
    pairs=[]
    for qid, values in grouped.items():
        if 0 not in values:
            continue
        _, residual=values[0]
        for index,(job, challenger) in values.items():
            if index == 0:
                continue
            ll=compare_evidence(residual,challenger,'ll_expansions')
            hl=compare_evidence(residual,challenger,'nodes_expanded')
            x=np.asarray(job['query']['features'])
            pairs.append(dict(pair_id=f'{qid}-{index}', query_id=qid,
                family_id=job['instance']['family_id'], split=job['instance']['split'],
                feature_difference=(x[index]-x[0]).tolist(), ll_label=ll['label'],hl_label=hl['label'],
                ll_evidence=ll,hl_evidence=hl))
    schedule=read(ROOT/'replay_schedule.json')
    expected={j['job_id'] for j in schedule}
    actual={j['job_id'] for j,r in jobs_outputs(ROOT/'replays')}
    report=dict(created_utc=utc(), scheduled=len(expected), attempted=len(actual),
                missing_jobs=sorted(expected-actual), errors=errors, statuses=dict(statuses),
                queries=len(grouped), pairs=len(pairs),
                ll_labels=dict(Counter(str(p['ll_label']) for p in pairs)),
                hl_only_pairs=sum(p['ll_label'] is None and p['hl_label'] is not None for p in pairs))
    write_new(ROOT/'pairs.json',pairs)
    write_new(ROOT/'audit.json',report)
    if report['missing_jobs'] or errors:
        raise ValueError('Incomplete or failed raw data; inspect audit before proceeding')
    return pairs


def fit():
    from mapf.multisignal_model import fit_pairwise, save_model
    from mapf.multisignal_teacher import FEATURE_NAMES
    pairs=pairs_and_audit()
    for variant in ('confirmed','confirmed_aux'):
        model=fit_pairwise([p for p in pairs if p['split']=='train'],FEATURE_NAMES,variant=variant,
                          provenance=dict(manifest_sha256=sha(ROOT/'manifest.json'),
                                          pairs_sha256=sha(ROOT/'pairs.json'),source=source_manifest()))
        save_model(ROOT/'models'/(variant+'.unselected.json'),model)
        print(variant, model['trained'],model['confirmed_class_counts'],flush=True)


def rows(directory, expected):
    observed={j['job_id']:(j,r) for j,r in jobs_outputs(directory)}
    result=[]
    costs=defaultdict(set)
    for job in expected:
        _, output=observed.get(job['job_id'],(job,None))
        data=(output or {}).get('data',{})
        solved=output is not None and output.get('status')=='OK' and data.get('status')=='SOLVED' and data.get('solution_validated')
        if solved:
            costs[job['instance']['id']].add(data['cost'])
        result.append(dict(job_id=job['job_id'], family_id=job['instance']['family_id'],
            kind=job['instance']['kind'],policy=job['policy'],tau=job.get('tau'),
            solved=bool(solved),status=data.get('status','MISSING_OR_ERROR'),
            capped_seconds=min(job['timeout'],data['runtime']) if solved else job['timeout'],
            instance_id=job['instance']['id']))
    if any(len(v)>1 for v in costs.values()):
        raise ValueError('Optimal cost disagreement across policies')
    return result


def family_ratio(values, policy, baseline='residual'):
    groups=defaultdict(lambda:defaultdict(list))
    kinds={}
    for row in values:
        groups[row['family_id']][row['policy']].append(row['capped_seconds'])
        kinds[row['family_id']]=row['kind']
    ratios={f:statistics.mean(g[policy])/statistics.mean(g[baseline]) for f,g in groups.items()
            if policy in g and baseline in g}
    if not ratios:
        raise ValueError('No paired families')
    estimate=math.exp(statistics.mean(math.log(v) for v in ratios.values()))
    rng=np.random.default_rng(SEED)
    strata=[[math.log(ratios[f]) for f in ratios if kinds[f]==kind] for kind in ('even','random')]
    samples=np.zeros(10000)
    for values_in_stratum in strata:
        if values_in_stratum:
            samples+=rng.choice(values_in_stratum,(10000,len(values_in_stratum)),replace=True).sum(axis=1)
    samples=np.exp(samples/len(ratios))
    return dict(ratio=estimate,ci95=np.quantile(samples,[.025,.975]).tolist(),families=len(ratios))


def validate():
    from mapf.multisignal_model import THRESHOLDS,set_threshold,save_model
    manifest,design=read(ROOT/'manifest.json'),read(ROOT/'design.json')
    instances=[i for i in manifest['instances'] if i['split']=='validation' and i['agents'] in design['selected_agent_counts']]
    source=source_manifest()
    jobs=[make_job('solve',i,'residual',10.,source) for i in instances]
    models={v:read(ROOT/'models'/(v+'.unselected.json')) for v in ('confirmed','confirmed_aux')}
    for variant,model in models.items():
        if model['trained']:
            jobs += [make_job('solve',i,variant,10.,source,tau=tau,
                             model_path=(ROOT/'models'/(variant+'.unselected.json')).as_posix(),
                             model_sha256=sha(ROOT/'models'/(variant+'.unselected.json')))
                     for i in instances for tau in THRESHOLDS]
    random.Random(SEED+2).shuffle(jobs)
    write_new(ROOT/'validation_schedule.json',jobs)
    run_jobs(jobs,ROOT/'validation',1,3600)
    measurements=rows(ROOT/'validation',jobs)
    if any(r['status']=='MISSING_OR_ERROR' for r in measurements):
        raise ValueError('Validation incomplete')
    selections={}
    pairs=read(ROOT/'pairs.json')
    support={p['family_id'] for p in pairs if p['split']=='validation' and p['ll_label']==1}
    for variant,model in models.items():
        tau=1.01
        diagnostics=[]
        if model['trained'] and len(support)>=3:
            baseline=[r for r in measurements if r['policy']=='residual']
            for threshold in THRESHOLDS:
                subset=[r for r in measurements if r['policy']==variant and r['tau']==threshold]
                ratio=family_ratio(baseline+subset,variant)
                diagnostics.append(dict(tau=threshold,**ratio,solved=sum(r['solved'] for r in subset)))
            eligible=[d for d in diagnostics if d['ratio']<1 and d['solved']>=sum(r['solved'] for r in baseline)]
            if eligible:
                tau=min(eligible,key=lambda d:(d['ratio'],-d['tau']))['tau']
        provenance=dict(informative_validation_families=len(support),diagnostics=diagnostics,
                        rule='min family ratio with no solve loss; >=3 validation families with LL wins; else disabled',
                        schedule_sha256=sha(ROOT/'validation_schedule.json'))
        save_model(ROOT/'models'/(variant+'.json'),set_threshold(model,tau,provenance))
        selections[variant]=dict(tau=tau,trained=model['trained'],**provenance)
    write_new(ROOT/'validation_selection.json',selections)


def benchmark():
    manifest,design=read(ROOT/'manifest.json'),read(ROOT/'design.json')
    source=source_manifest()
    policies=['plain','residual','pool_random','model_b','confirmed','confirmed_aux']
    instances=[i for i in manifest['instances'] if i['split']=='test' and i['agents'] in design['selected_agent_counts']]
    jobs=[]
    for i in instances:
        for policy in policies:
            for repeat in range(2):
                extra=dict(repeat=repeat)
                if policy in ('confirmed','confirmed_aux'):
                    path=ROOT/'models'/(policy+'.json')
                    extra.update(model_path=path.as_posix(),model_sha256=sha(path),tau=read(path)['tau'])
                jobs.append(make_job('solve',i,policy,20.,source,**extra))
    random.Random(SEED+3).shuffle(jobs)
    write_new(ROOT/'test_schedule.json',jobs)
    with zipfile.ZipFile(ROOT/'frozen_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in source:
            archive.write(path)
    write_new(ROOT/'test_freeze.json',dict(created_utc=utc(),source=source,
        archive_sha256=sha(ROOT/'frozen_source.zip'),schedule_sha256=sha(ROOT/'test_schedule.json'),
        primary='confirmed_aux versus residual; family geometric capped-runtime ratio',
        required_success='upper CI95 < 1; no fewer solves; complete valid schedule'))
    run_jobs(jobs,ROOT/'test',1,7140)
    measurements=rows(ROOT/'test',jobs)
    comparisons={p:family_ratio(measurements,p) for p in policies if p!='residual'}
    solved={p:sum(r['solved'] for r in measurements if r['policy']==p) for p in policies}
    missing=sum(r['status']=='MISSING_OR_ERROR' for r in measurements)
    supported=(not missing and comparisons['confirmed_aux']['ci95'][1]<1 and solved['confirmed_aux']>=solved['residual'])
    summary=dict(comparisons=comparisons,solved=solved,missing_or_error=missing,
                 primary_speed_claim_supported=supported,
                 auxiliary_ablation=family_ratio(measurements,'confirmed_aux','confirmed'))
    write_new(ROOT/'test_rows.json',measurements)
    write_new(ROOT/'summary.json',summary)
    lines=['# MS8 measured results','',f'Primary runtime improvement supported: **{supported}**.',
           f'Missing/error runs: {missing}. This is one fixed empty map, not a state-of-the-art comparison.','',
           '| Policy | Solved | Runtime ratio vs residual | 95% interval |',
           '|---|---:|---:|---|']
    for p in policies:
        c=comparisons.get(p,dict(ratio=1.,ci95=[1.,1.]))
        lines.append(f"| {p} | {solved[p]}/48 | {c['ratio']:.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] |")
    lines += ['', 'Ratios aggregate two counts and two repeats within each of 12 scenario files.',
              'Failed solves receive the full cap. Intervals bootstrap files within even/random strata.',
              'The implemented teacher uses LL work and optional HL evidence. D_phi, horizon fusion and DAgger are not implemented in this stage.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    commands={'fit':fit,'validate':validate,'benchmark':benchmark}
    commands[sys.argv[1]]()
