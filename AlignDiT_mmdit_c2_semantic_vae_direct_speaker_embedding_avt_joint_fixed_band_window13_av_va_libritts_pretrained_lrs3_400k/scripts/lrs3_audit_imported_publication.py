"""Independently reconcile imported LRS3 publications with their pinned origin.

Reads manifests, text evidence metadata, and quality reports without rehashing
all WAV/ASR evidence or model-feature arrays. Never modifies a publication.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

FREEZE_SHA = '89791cdc221086a3f592d2bc657c9d0041f0570b198cfa5e239acc028f54d892'
MEDIA_TERMINAL_SHA = '7d5034439fe1df002a79b8c707ff0f6cebd96ef85ebca3e17847e8363c8dadb0'
PRIOR_TEXT_AUDIT_SHA = 'a859ac312a4bb776523ed327a320204f6ec9f543cd0db522a44ad56a824a4dbc'
PINNED_BASELINE = {'full': 292405, 'full_pseudo': 293164}
ISSUES = []
HASHES = {}


def check(ok, message, detail=None):
    if not ok:
        ISSUES.append({'message': message, 'detail': detail})
        if len(ISSUES) <= 15:
            print(json.dumps({'audit_issue': ISSUES[-1]}, ensure_ascii=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def jread(path):
    p = Path(path)
    b = p.read_bytes()
    HASHES[str(p)] = hashlib.sha256(b).hexdigest()
    return json.loads(b)


def rows(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for line in f:
            h.update(line)
            if line.strip():
                yield json.loads(line)
    HASHES[str(p)] = h.hexdigest()


def clean(s):
    return ' '.join(re.sub(r'\{(?:lg|ns|pc|tc)\}', ' ', s, flags=re.I).lower().split())


def rowhash(r):
    return hashlib.sha256(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def approx(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)


def project_eligibility(source, labels, partitions, vocabulary, media, *, include_p1):
    """Derive admissible IDs from source/text policy, independently of caches."""
    expected, baseline = {}, set()
    for sid, row in source.items():
        label = labels.get(sid)
        promoted = not row['text'] and (
            sid in partitions['strict'] or include_p1 and sid in partitions['p1'])
        text = clean(label['text']) if promoted else row['text']
        kind = (label['source_kind'] if promoted else
                'existing_transcript' if row['text'] else 'unresolved')
        status = (('asr_pseudo' if kind == 'asr_pseudo' else 'recovered_consensus')
                  if promoted else row['text_status'])
        reasons = []
        if not text:
            reasons.append('text_unresolved')
        if set(text) - vocabulary:
            reasons.append('text_oov')
        if not row['duration_eligible']:
            reasons.append('duration_outside_0.4_30_seconds')
        if abs(row['audio_seconds'] - row['video_seconds']) > .10:
            reasons.append('audio_video_duration_delta')
        if kind == 'asr_pseudo' and row['split'] != 'train':
            reasons.append('pseudo_outside_train')
        if not reasons:
            baseline.add(sid)
        if sid in media:
            reasons.append('media_quality:' + media[sid]['reason'])
        expected[sid] = (text, kind, status, reasons, promoted)
    return expected, baseline


def readable_report(summary):
    root, origin = Path(summary['root']), Path(summary['origin_root'])
    lines = [
        'LRS3 导入数据发布：独立文本与覆盖审计',
        '审计结果：' + ('PASS' if summary['passed'] else 'FAIL，请先处理 summary.json 中的 issues'),
        f"新数据根目录：{root}", f"只读来源根目录：{origin}",
        f"来源覆盖：{summary['canonical_source_pairs']} 个片段 / {summary['canonical_original_ids']} 个原始 ID",
        f"来源音频总时长：{sum(summary['canonical_source_audio_hours_by_split'].values()):.6f} 小时（按实际片段相加）",
        f"固定媒体质量拒绝：{summary['media_quality_excluded_ids']} 个片段", '',
    ]
    for mode, title in [('full', '默认 strict'), ('full_pseudo', '仅训练集增加 P1 的扩展')]:
        p = summary['publications'][mode]
        lines.append(f"{title}：{p['published_ready']} 条可用；原始 ID 报告覆盖 {p['original_ids_accounted']} 条。")
        for split in ('train', 'valid', 'test'):
            lines.append(f"  {split}: {p['split_counts'][split]} 条，{p['audio_hours_by_split'].get(split,0):.6f} 小时")
        lines.append(f"  接纳集合：基线 {p['base_eligible_before_media']} 减去与媒体拒绝交集 {p['media_quality_overlap_with_base']}")
        if mode == 'full_pseudo':
            lines.append(f"  相比 strict 新增 {p['added_train_only_p1']} 条 train P1；valid/test 字节完全一致：{summary['evaluation_manifest_byte_identical']}")
        lines.append(f"  发布目录：{root/'manifests'/mode}")
        lines.append(f"  逐原始 ID 报告：{root/'reports'/f'{mode}_original_coverage.jsonl'}")
        lines.append('')
    manual = summary['manual_review']
    controls = summary['prior_independent_text_quality_audit']['controls']
    lines += [
        f"尚未批准的文本：{manual['segments']} 条 / {manual['original_ids']} 个原始 ID / {manual['hours']:.6f} 小时。",
        f"人工队列：{origin/'text_recovery/manual_review/segments_for_human_review.jsonl'}",
        f"复核说明：{origin/'text_recovery/manual_review/REVIEW_INSTRUCTIONS.txt'}",
        f"文本来源：{summary['frozen_release']}",
        f"有效文本冻结 SHA256：{summary['frozen_export_sha256']}",
        f"导入来源契约：{root/'reports/imported_source.json'}",
        f"全量导入完成凭据：{root/'reports/sharded_import/completion.json'}",
        f"媒体质量证据：{origin/'reports/media_full_pipeline/quality_terminal.json'}",
        f"逐恢复片段发布对账：{summary['recovery_accounting_path']}",
        f"详细审计与输入文件 SHA：{root/'reports/text_publication_independent_audit/summary.json'}", '',
        f"已知文本对照：{controls['exact_matches']}/{controls['count']} 完全一致，corpus WER={controls['word_edits']}/{controls['reference_words']}={controls['corpus_wer']:.8f}。",
        '自动 source consensus 与 P1 均不等于原始人工真值；对照 WER 不是恢复子集的错误率。',
        '原始 ID 全部记账不等于所有片段均可训练；可训练时长受文本、时长、词表和媒体质量条件限制。',
        '本次只读对账复用此前已完成的 WAV/ASR 证据审计，不重复全量特征或证据哈希，也不证明音画同步。',
    ]
    return '\n'.join(lines) + '\n'


def audit(root, origin_root, *, vocabulary_path=None, expected_import_sha256=None):
    from aligndit.lrs3_imported_source import (
        validate_imported_source, imported_source_scope, validate_import_completion,
    )

    root = Path(root).resolve(strict=True)
    origin_root = Path(origin_root).resolve(strict=True)
    if root == origin_root:
        raise ValueError("Imported publication root must differ from its origin")
    out = root / "reports/text_publication_independent_audit"
    release = origin_root / "text_recovery/release_v2"
    ISSUES.clear()
    HASHES.clear()
    started = time.time()
    # The completion marker is written last by the producer. Never inspect a partial publication.
    for mode in ('full', 'full_pseudo'):
        if not (root / 'manifests' / mode / 'cache_completion.json').exists():
            print(json.dumps({'waiting_for': mode + '/cache_completion.json'}), flush=True)
            return 3
    imported = validate_imported_source(root, expected_sha256=expected_import_sha256)
    if imported is None:
        raise ValueError('The destination has no imported-source provenance contract')
    if Path(imported['origin_root']).resolve() != origin_root:
        raise ValueError('Explicit origin root differs from pinned imported source')
    import_scope = imported_source_scope(imported)
    HASHES[str(imported['_path'])] = imported['_sha256']
    out.mkdir(exist_ok=True, parents=True)
    receipts = {m: jread(root/'manifests'/m/'cache_completion.json') for m in PINNED_BASELINE}
    contract = jread(root/'reports/source_contract.json')
    freeze = jread(release/'frozen_export.json')
    prior_path=origin_root/'reports/text_recovery_final_quality/summary.json'
    prior=jread(prior_path)
    check(HASHES[str(prior_path)] == PRIOR_TEXT_AUDIT_SHA,
          'Previously completed WAV/ASR evidence audit receipt changed')
    check(prior.get('audit_complete') is True and prior.get('frozen_export_sha256')==FREEZE_SHA,
          'Prior independent recovery audit does not bind this release')
    check(HASHES[str(release/'frozen_export.json')] == FREEZE_SHA, 'Frozen release identity differs')
    required = {'coverage_by_original.jsonl', 'labels.excluded.jsonl', 'labels.pending.jsonl',
                'labels.pseudo_p1.accepted.jsonl', 'labels.pseudo_p2.candidates.jsonl',
                'labels.strict.accepted.jsonl', 'rules.json', 'summary.json'}
    check(required <= set(freeze.get('files', {})), 'Frozen export omits required files')
    for name in required:
        value = sha(release/name)
        HASHES[str(release/name)] = value
        check(value == freeze.get('files', {}).get(name), 'Frozen output hash differs', name)
    partition_names = {'strict':'labels.strict.accepted.jsonl', 'p1':'labels.pseudo_p1.accepted.jsonl',
                       'p2':'labels.pseudo_p2.candidates.jsonl', 'excluded':'labels.excluded.jsonl'}
    labels = {}
    partitions = {}
    for part, filename in partition_names.items():
        ids = set()
        for r in rows(release/filename):
            sid = r['sample_id']
            check(sid not in labels, 'Recovery partitions overlap', sid)
            labels[sid] = r
            ids.add(sid)
            check(r.get('accepted') is (part in ('strict','p1')), 'Recovery accepted flag differs', sid)
        partitions[part] = ids
    check([len(partitions[p]) for p in partition_names] == [11437,759,1613,7321], 'Recovery counts differ')

    source = {}
    originals = collections.defaultdict(list)
    keep = ['original_clip_id','split','text','text_status','label_source','label_validation',
            'audio_seconds','video_seconds','duration_eligible','relative_stem','utterance_key']
    for r in rows(root/'manifests/source_inventory.jsonl'):
        sid = r['sample_id']
        check(sid not in source, 'Duplicate canonical sample ID', sid)
        source[sid] = {k:r[k] for k in keep}
        originals[r['original_clip_id']].append(sid)
    check(HASHES[str(root/'manifests/source_inventory.jsonl')] == contract['source_inventory_sha256'] == freeze['source_manifest_sha256'], 'Canonical source hash differs')
    check(len(source) == 302167 and len(originals) == 151819, 'Canonical scope differs')
    # Reuse the canonical IDs already read above. This validates the full merge
    # ledger once; it does not reread individual feature/audio assets.
    imported_cache = validate_import_completion(root, imported, (
        {'sample_id': sid, 'relative_stem': row['relative_stem']}
        for sid, row in source.items()
    ))
    for path in imported_cache['artifacts']:
        HASHES[str(path)] = sha(path)
    missing = {s for s,r in source.items() if not r['text']}
    check(set(labels) == missing, 'Recovery partition does not exactly cover missing text')
    for sid, label in labels.items():
        check(label['original_clip_id'] == source[sid]['original_clip_id'], 'Recovery original identity differs', sid)
    if vocabulary_path is None:
        candidates = [Path(p) for p, fingerprint in contract.get('input_sha256', {}).items()
                      if fingerprint == contract['vocab_sha256'] and Path(p).name == 'vocab.txt']
        if len(candidates) != 1:
            raise ValueError('Cannot infer unique vocabulary; pass --vocab')
        vocabulary_path = candidates[0]
    vocabulary_path = Path(vocabulary_path).resolve(strict=True)
    vocabulary = set(vocabulary_path.read_text().splitlines())
    check(sha(vocabulary_path) == contract['vocab_sha256'], 'Vocabulary hash differs')

    terminal_path = origin_root/'reports/media_full_pipeline/quality_terminal.json'
    terminal = jread(terminal_path) if terminal_path.exists() else None
    check(terminal is not None and HASHES.get(str(terminal_path)) == MEDIA_TERMINAL_SHA,
          'Expected sealed media-quality terminal identity differs')
    media = {}
    if terminal:
        exclusion_path=origin_root/'reports/media_full_pipeline/media_quality_exclusions.json'
        exclusion_doc=jread(exclusion_path)
        refs={x['path']:x['sha256'] for x in terminal['evidence_files']}
        check(refs.get(str(exclusion_path))==HASHES[str(exclusion_path)], 'Media exclusion bytes differ from sealed terminal')
        check(terminal.get('source_manifest_sha256')==contract['source_inventory_sha256'] and terminal.get('source_contract_sha256')==HASHES[str(root/'reports/source_contract.json')], 'Media terminal binds different source scope')
        check(terminal.get('processing_complete') is True and terminal.get('all_source_ids_accounted') is True, 'Media terminal is not fully accounted')
        media={r['sample_id']:r for r in exclusion_doc['excluded']}
        check(len(media)==len(exclusion_doc['excluded']), 'Duplicate media exclusions')
    check(set(media) <= set(source), 'Media quality contains unknown IDs')
    if terminal:
        check(len(media) == terminal['excluded_count'], 'Media exclusion count differs')
        check(terminal['requested'] == len(source), 'Media terminal source scope differs')
    publication = {}
    strict_hashes = {}
    export_rows = []
    report_counter = {}
    for mode in ('full', 'full_pseudo'):
        is_pseudo = mode == 'full_pseudo'
        expected, base_ids = project_eligibility(
            source, labels, partitions, vocabulary, media, include_p1=is_pseudo)
        check(len(base_ids) == PINNED_BASELINE[mode], 'Readiness before final media exclusions differs from pinned baseline', {'mode':mode,'actual':len(base_ids),'expected':PINNED_BASELINE[mode]})
        wanted = base_ids - media.keys()
        manifest_dir = root/'manifests'/mode
        meta = jread(manifest_dir/'inventory_meta.json')
        policy = jread(manifest_dir/'text_policy.json')
        readiness = jread(root/f'reports/{mode}_readiness.json')
        receipt = receipts[mode]
        psha = HASHES[str(manifest_dir/'text_policy.json')]
        check(meta.get('imported_source') == receipt.get('imported_source') == readiness.get('imported_source') == import_scope,
              'Publication imported-source provenance differs from the validated contract', mode)
        check(meta.get('imported_cache') == receipt.get('imported_cache') == readiness.get('imported_cache') == imported_cache['scope'],
              'Publication imported-cache provenance differs from the full merge ledger', mode)
        policy_name = 'source_plus_asr_pseudo' if is_pseudo else 'source_grounded'
        check(meta.get('text_policy') == policy.get('policy') == receipt.get('text_policy') == readiness.get('text_policy') == policy_name, 'Published policy differs', mode)
        check(meta['text_policy_sha256'] == receipt['text_policy_sha256'] == psha, 'Policy hash differs', mode)
        check(receipt.get('status') == meta.get('status') == 'complete', 'Incomplete publication', mode)
        check(meta['source_original_ids'] == receipt['source_original_ids'] == 151819 and meta['source_pairs'] == 302167, 'Published source scope differs', mode)
        if terminal:
            check(meta.get('media_quality')==receipt.get('media_quality')==readiness.get('media_quality'), 'Publication media-quality scopes differ', mode)
            check(meta['media_quality']['terminal_sha256']==HASHES[str(terminal_path)] and meta['media_quality']['quality_excluded_pairs']==len(media), 'Publication uses a different media-quality terminal',mode)
        expected_files = {partition_names['strict']} | ({partition_names['p1']} if is_pseudo else set())
        actual_pins = policy.get('recovered_label_files', [])
        check({x['path'] for x in actual_pins} == {str(release/f) for f in expected_files}, 'Publication label paths differ from release_v2', mode)
        for pin in actual_pins:
            check(pin['sha256'] == freeze['files'].get(Path(pin['path']).name), 'Publication label hash differs', pin)
        pins = readiness.get('publication_input_pins',{})
        check(pins.get(str(release/'frozen_export.json')) == FREEZE_SHA, 'Readiness does not bind expected freeze', mode)
        check(not any('/text_recovery/release/' in k for k in pins), 'Readiness references invalid original release', mode)

        accepted = {}
        kindcounts = {s:collections.Counter() for s in ('train','valid','test')}
        hours = collections.Counter()
        for r in rows(manifest_dir/'inventory.jsonl'):
            sid=r['sample_id']; sr=source.get(sid)
            check(sr is not None and sid not in accepted, 'Unknown or duplicate published ID', {'mode':mode,'id':sid})
            if sr is None: continue
            import_entry = imported_cache['receipts'].get(sid, {})
            check(import_entry.get('relative_stem') == r['relative_stem']
                  and import_entry.get('import_receipt_sha256') == r.get('import_receipt_sha256'),
                  'Published import receipt differs from the pinned ledger', {'mode': mode, 'id': sid})
            txt,kind,status,reasons,promoted=expected[sid]
            check(not reasons, 'Published ID fails independent eligibility', {'mode':mode,'id':sid,'reasons':reasons})
            check(all(r[k]==sr[k] for k in ['original_clip_id','split','relative_stem','utterance_key','audio_seconds','video_seconds']), 'Published identity differs', sid)
            check((r['text'],r['source_kind'],r['text_status']) == (txt,kind,status), 'Published text differs', {'mode':mode,'id':sid})
            if promoted:
                check(r.get('recovery_evidence') == labels[sid], 'Embedded recovery evidence differs from frozen label', sid)
                check(r.get('independent_content_verified') is False, 'Automatic text incorrectly marked independently verified', sid)
            if kind=='asr_pseudo':check(is_pseudo and r['split']=='train' and sid in partitions['p1'], 'Pseudo label outside allowed training extension', sid)
            rh = rowhash(r);accepted[sid]=(rh,r['split'],kind)
            if is_pseudo and sid in strict_hashes:check(rh==strict_hashes[sid][0], 'Pseudo extension changes strict row', sid)
            kindcounts[r['split']][kind]+=1;hours[r['split']]+=r['audio_seconds']/3600
        missing_ready = wanted - accepted.keys()
        extra_ready = accepted.keys() - wanted
        check(not missing_ready and not extra_ready, 'Final readiness set differs from baseline minus media quality exclusions', {'mode':mode,'missing_count':len(missing_ready),'unexpected_count':len(extra_ready),'missing_examples':sorted(missing_ready)[:30],'unexpected_examples':sorted(extra_ready)[:30]})
        check(len(accepted) == meta['training_ready_pairs'] == receipt['count'] == readiness['ready_pairs'], 'Publication readiness count differs', mode)
        check(HASHES[str(manifest_dir/'inventory.jsonl')] == receipt['inventory_sha256'] == meta['manifests']['inventory.jsonl']['sha256'], 'Inventory SHA differs', mode)
        for split in ('train','valid','test'):
            seen=set();p=manifest_dir/f'{split}.jsonl'
            for r in rows(p):
                sid=r['sample_id'];check(sid not in seen and sid in accepted and r['split']==split and rowhash(r)==accepted[sid][0], 'Split row differs from inventory', {'mode':mode,'split':split,'id':sid});seen.add(sid)
                if split!='train':check(r['source_kind']!='asr_pseudo' and r['text_status']!='asr_pseudo', 'Evaluation contains pseudo label', sid)
            check(seen=={s for s,v in accepted.items() if v[1]==split}, 'Split membership differs', {'mode':mode,'split':split})
            check(HASHES[str(p)]==meta['manifests'][p.name]['sha256'], 'Split SHA differs', str(p))
            if is_pseudo and split!='train':check(HASHES[str(p)]==HASHES[str(root/'manifests/full'/p.name)], 'Strict and pseudo evaluation are not byte-identical', split)

        quality={};reasons_count=collections.Counter();nonmedia_extra=[]
        quality_path=root/f'reports/{mode}_sample_quality.jsonl'
        for q in rows(quality_path):
            sid=q['sample_id'];check(sid in source and sid not in quality, 'Quality scope ID duplicate/unknown', sid)
            if sid not in source:continue
            txt,kind,status,wanted_reasons,promoted=expected[sid]
            check((q['text'],q['source_kind'],q['text_status'])==(txt,kind,status), 'Quality text differs', {'mode':mode,'id':sid})
            check(q['ready']==(sid in accepted) and bool(q['reasons'])!=q['ready'], 'Quality readiness differs', sid)
            check(set(wanted_reasons)<=set(q['reasons']), 'Quality report omits independently expected exclusion', {'mode':mode,'id':sid})
            extra=set(q['reasons'])-set(wanted_reasons)
            if extra and sid not in media:nonmedia_extra.append({'sample_id':sid,'reasons':sorted(extra)})
            if sid in media:check(q.get('media_quality_exclusion')==media[sid], 'Quality media exclusion does not bind terminal row', sid)
            cache_ready=len(q['cache_sha256'])==4 and not any('provenance:' in x or x.startswith('media_quality:') for x in q['reasons'])
            quality[sid]=(q['ready'],q['reasons'],q['text_status'],q['source_kind'],q['ctc_feasible_40hz'],q['media_preprocessing']['crop_kind'] or 'unknown',cache_ready)
            reasons_count.update(q['reasons'])
        check(set(quality)==set(source), 'Quality does not account for all 302167 source IDs', mode)
        check(not nonmedia_extra, 'Unexpected non-media exclusion reasons require investigation', {'mode':mode,'count':len(nonmedia_extra),'examples':nonmedia_extra[:30]})
        check(dict(reasons_count)==readiness['pending_reasons'], 'Readiness reason counts differ', mode)
        original_seen=set();fully=0;original_tsv_hashes={}
        tsv_fields=['original_clip_id','split','source_pairs','ready_pairs','cache_ready_pairs',
                    'source_audio_seconds','ready_audio_seconds','ready_duration_fraction',
                    'fully_covered','ctc_infeasible_pairs','text_status_counts','source_kind_counts',
                    'crop_kind_counts','pending_reasons']
        for o in rows(root/f'reports/{mode}_original_coverage.jsonl'):
            oid=o['original_clip_id'];check(oid in originals and oid not in original_seen, 'Original report duplicate/unknown', oid);original_seen.add(oid)
            if oid not in originals:continue
            members=originals[oid];ready=[s for s in members if s in accepted]
            samplemap={x['sample_id']:x for x in o['samples']}
            check(len(samplemap)==len(o['samples']) and set(samplemap)==set(members), 'Original member IDs differ', oid)
            check(o['source_pairs']==len(members) and o['ready_pairs']==len(ready), 'Original counts differ', oid)
            check(o['cache_ready_pairs']==sum(quality[s][6] for s in members) and o['ctc_infeasible_pairs']==sum(not quality[s][4] for s in members), 'Original cache/CTC aggregate differs', oid)
            check(approx(o['source_audio_seconds'],sum(source[s]['audio_seconds'] for s in members)) and approx(o['ready_audio_seconds'],sum(source[s]['audio_seconds'] for s in ready)), 'Original duration accounting differs', oid)
            expected_counts={'pending_reasons':collections.Counter(x for s in members for x in quality[s][1]),'text_status_counts':collections.Counter(quality[s][2] for s in members),'source_kind_counts':collections.Counter(quality[s][3] for s in members),'crop_kind_counts':collections.Counter(quality[s][5] for s in members)}
            for field,count in expected_counts.items():check(dict(count)==o[field], 'Original aggregate differs', {'original':oid,'field':field})
            for sid,x in samplemap.items():check(x['ready']==quality[sid][0] and x['reasons']==quality[sid][1], 'Original/sample-quality readiness differs', sid)
            check(o['fully_covered']==(len(ready)==len(members)), 'Original fully-covered flag differs', oid)
            original_tsv_hashes[oid]=rowhash({k:o[k] for k in tsv_fields})
            fully+=o['fully_covered']
        check(original_seen==set(originals), 'Original coverage omits canonical original IDs', mode)
        check(fully==readiness['fully_covered_original_ids'], 'Fully-covered original aggregate differs', mode)
        tsv_path=root/f'reports/{mode}_original_coverage.tsv';tsv_seen=set()
        with tsv_path.open(newline='') as f:
            reader=csv.DictReader(f,delimiter='\t')
            check(reader.fieldnames==tsv_fields,'Original coverage TSV fields differ',mode)
            for t in reader:
                oid=t['original_clip_id'];check(oid not in tsv_seen,'Original TSV contains duplicates',oid);tsv_seen.add(oid)
                for k in ['source_pairs','ready_pairs','cache_ready_pairs','ctc_infeasible_pairs']:t[k]=int(t[k])
                for k in ['source_audio_seconds','ready_audio_seconds','ready_duration_fraction']:t[k]=float(t[k])
                check(t['fully_covered'] in {'True','False'},'Invalid TSV boolean',oid)
                t['fully_covered']=t['fully_covered']=='True'
                for k in ['text_status_counts','source_kind_counts','crop_kind_counts','pending_reasons']:t[k]=json.loads(t[k])
                check(rowhash(t)==original_tsv_hashes.get(oid),'Original TSV differs from audited JSONL',oid)
        HASHES[str(tsv_path)]=sha(tsv_path)
        check(tsv_seen==original_seen,'Original TSV scope differs',mode)
        pub={'base_eligible_before_media':len(base_ids),'media_quality_overlap_with_base':len(base_ids & media.keys()),'expected_final_ready':len(wanted),'published_ready':len(accepted),'excluded_pairs':len(source)-len(accepted),'original_ids_accounted':len(original_seen),'fully_covered_original_ids':fully,'split_source_kind_counts':{s:dict(c) for s,c in kindcounts.items()},'audio_hours_by_split':dict(hours),'pending_reason_counts':dict(reasons_count),'unexpected_nonmedia_extra_reasons':nonmedia_extra}
        pub['split_counts'] = {split: sum(counts.values()) for split, counts in kindcounts.items()}
        pub['accepted_id_sha256'] = hashlib.sha256(('\n'.join(sorted(accepted))+'\n').encode()).hexdigest()
        pub['accepted_id_hash_serialization'] = 'UTF-8 sorted sample IDs, newline-terminated'
        publication[mode]=pub
        for part,ids in partitions.items():
            part_ready=ids & accepted.keys()
            pub.setdefault('recovery_partition_accounting',{})[part]={'released_segments':len(ids),'published':len(part_ready),'excluded':len(ids)-len(part_ready),'excluded_reasons':dict(collections.Counter(x for s in ids-accepted.keys() for x in quality[s][1]))}
            for sid in sorted(ids):
                export_rows.append({'publication':mode,'sample_id':sid,'original_clip_id':source[sid]['original_clip_id'],'recovery_partition':part,'published':sid in accepted,'reasons':quality[sid][1]})
        if is_pseudo:
            added=accepted.keys()-strict_hashes.keys()
            check(strict_hashes.keys()<=accepted.keys(), 'Pseudo extension removed strict IDs')
            check(added == partitions['p1'] & wanted, 'Pseudo extension added IDs differ from eligible P1')
            pub['added_train_only_p1']=len(added)
            baseline=policy.get('strict_baseline',{})
            for f in ('inventory','train','valid','test'):
                check(baseline.get(f+'_sha256')==HASHES[str(root/'manifests/full'/(f+'.jsonl'))], 'Pseudo policy strict-baseline hash differs', f)
        else:strict_hashes=accepted
        report_counter[mode]=set(accepted)
        print(json.dumps({'audited_publication':mode,**{k:v for k,v in pub.items() if k not in ('pending_reason_counts','recovery_partition_accounting','split_source_kind_counts')}},ensure_ascii=False),flush=True)

    manual=origin_root/'text_recovery/manual_review'
    manual_summary=jread(manual/'summary.json')
    check(manual_summary['frozen_export_sha256']==FREEZE_SHA and manual_summary['release_directory']==str(release), 'Manual review does not bind valid release_v2')
    seen=set();review_originals=collections.defaultdict(set);review_hours=0
    for r in rows(manual/'segments_for_human_review.jsonl'):
        sid=r['sample_id'];check(sid not in seen, 'Manual queue duplicate', sid);seen.add(sid)
        check(r.get('accepted') is False and r.get('training_eligible') is False and r.get('human_annotation_status')=='not_reviewed','Manual candidate accidentally accepted',sid)
        check(sid in source and r['original_clip_id']==source[sid]['original_clip_id'],'Manual original identity differs',sid)
        if sid in source:review_hours+=source[sid]['audio_seconds']/3600
        review_originals[r['original_clip_id']].add(sid)
        check(all(sid not in accepted_ids for accepted_ids in report_counter.values()),'Unreviewed manual candidate included in publication',sid)
    check(seen==partitions['p2']|partitions['excluded'] and len(seen)==8934, 'Manual queue differs from all remaining text candidates')
    original_queue=set()
    for r in rows(manual/'originals_with_remaining_review.jsonl'):
        oid=r['original_clip_id'];check(oid not in original_queue and set(r['sample_ids'])==review_originals[oid] and r['review_segments']==len(review_originals[oid]),'Manual original queue differs',oid);original_queue.add(oid)
    check(original_queue==set(review_originals) and len(original_queue)==3128,'Manual original count differs')
    check(approx(review_hours,manual_summary['hours_for_review']),'Manual duration accounting differs')

    # Recheck small completion receipts to prove the publications remained pinned throughout.
    for mode in PINNED_BASELINE:
        p=root/'manifests'/mode/'cache_completion.json'
        check(sha(p)==HASHES[str(p)],'Completion marker changed while auditing',mode)
    for p in [Path(imported['_path']), *imported_cache['artifacts']]:
        check(sha(p)==HASHES[str(p)],'Imported provenance changed while auditing',str(p))
    summary={'audit_complete':True,'passed':not ISSUES,'started_at':started,'finished_at':time.time(),'root':str(root),'origin_root':str(origin_root),'imported_source':import_scope,'frozen_release':str(release),'frozen_export_sha256':FREEZE_SHA,'canonical_source_pairs':len(source),'canonical_original_ids':len(originals),'media_quality_excluded_ids':len(media),'publications':publication,'manual_review':{'segments':len(seen),'original_ids':len(original_queue),'hours':review_hours,'accepted':False},'evaluation_manifest_byte_identical':all(HASHES[str(root/'manifests/full'/f'{s}.jsonl')]==HASHES[str(root/'manifests/full_pseudo'/f'{s}.jsonl')] for s in ('valid','test')),'prior_independent_text_quality_audit':{'path':str(prior_path),'sha256':HASHES[str(prior_path)],'controls':prior['controls']},'issues':ISSUES,'files_read_sha256':HASHES,'scope':'Independent text/set/split/original-coverage reconciliation. Does not rerun or change finalization, models, quality gates, source inputs, or caches.','limitations':['Previous independent audit already rehashed 21130 WAV and 42159 input evidence files; those expensive hashes were intentionally not repeated.','Automatic source consensus is not original or human-verified text. P1 remains a separate training-only pseudo-label extension.','Known-text control WER is not the estimated error rate of the filtered recovered-label subset.','Static frame review, cache dimensions and text audit do not establish audiovisual synchronization.']}
    summary['imported_cache'] = imported_cache['scope']
    summary['canonical_source_audio_hours_by_split'] = {
        split: math.fsum(row['audio_seconds'] for row in source.values() if row['split'] == split) / 3600
        for split in ('train', 'valid', 'test')
    }
    (out/'recovery_segment_publication_accounting.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in export_rows))
    summary['recovery_accounting_path']=str(out/'recovery_segment_publication_accounting.jsonl')
    summary['recovery_accounting_sha256']=sha(summary['recovery_accounting_path'])
    summary['audit_script_sha256']=sha(__file__)
    delivery_path = out/'DELIVERY_COUNTS.txt'
    delivery_path.write_text(readable_report(summary))
    summary['readable_report_path'] = str(delivery_path)
    summary['readable_report_sha256'] = sha(delivery_path)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'complete':True,'passed':not ISSUES,'issues':len(ISSUES),'report':str(out/'summary.json')},ensure_ascii=False),flush=True)
    return 0 if not ISSUES else 2


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Initialized imported cache root')
    parser.add_argument('--origin-root', type=Path, required=True, help='Read-only origin of media/text evidence')
    parser.add_argument('--vocab', type=Path, help='Optional vocabulary path override with identical pinned bytes')
    parser.add_argument('--expected-import-sha256', help='Optional explicit imported_source.json hash pin')
    args = parser.parse_args()
    sys.exit(audit(args.root, args.origin_root, vocabulary_path=args.vocab,
                   expected_import_sha256=args.expected_import_sha256))
