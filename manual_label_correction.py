#!/usr/bin/env python3
import csv, re, json
from collections import OrderedDict

mismatches_path = 'mismatches_for_manual_review.csv'
train_path = 'train_filled_corrected.csv'
log_path = 'correction_manual_log.csv'
final_json = 'final_verification.json'

# conservative regex helpers
re_tear = re.compile(r'\b(tear|tearing|teared|torn|rotura|rupture|ruptured|ruptura|yırtık|r?uptur|fissur|disruption|delamination|bucket[- ]handle|radial tear|root tear|root avulsion)\b', re.I)
re_partial = re.compile(r'\bpartial\b', re.I)
re_avulsion = re.compile(r'avulsion', re.I)

# label detectors
def detect_acl(text):
    if re.search(r'\b(acl|anterior cruciate)\b', text, re.I):
        # if explicit tear/rupture/avulsion/complete/partial
        if re_tear.search(text) or re_avulsion.search(text):
            return 1
    return 0

def detect_mcl(text):
    if re.search(r'\b(mcl|medial collateral|medial collateral ligament|ligament collat(?:eral) medial)\b', text, re.I):
        if re_tear.search(text) or re_avulsion.search(text):
            return 1
    return 0

def detect_medial_meniscus(text):
    # look for medial meniscus mentions with explicit tear words
    if re.search(r'medial menisc|menisco medial|menisco interno|menisco mediales|menisco méd', text, re.I):
        if re_tear.search(text) or re.search(r'bucket[- ]handle|extrud|extrusion|displaced fragment', text, re.I):
            return 1
        # degenerative tearing explicit
        if re.search(r'degenerat(?:ive)? tear|degenerate tearing|degenerat.*tearing', text, re.I):
            return 1
    return 0

def detect_lateral_meniscus(text):
    if re.search(r'lateral menisc|menisco lateral|menisco lat', text, re.I):
        if re_tear.search(text) or re.search(r'bucket[- ]handle|extrud|extrusion|displaced fragment', text, re.I):
            return 1
        if re.search(r'degenerat(?:ive)? tear|degenerate tearing|degenerat.*tearing', text, re.I):
            return 1
    return 0

def detect_oa(text, compartment):
    # compartment: 'medial','lateral','pf'
    patterns = [r'osteoarth', r'arthrosis', r'gonartro', r'OA', r'osteo-arth']
    comp_words = {
        'medial': [r'medial', r'medial compartment', r'femorotibial medial', r'medial femorotibial', r'medial compartment'],
        'lateral': [r'lateral', r'lateral compartment', r'femorotibial lateral', r'lateral femorotibial'],
        'pf': [r'patellofemoral', r'patellofemoral joint', r'patellofemoral joint', r'patellofemoral articular', r'patellofemoral', r'patellofemoral']
    }
    textlow = text.lower()
    found_comp = any(re.search(pat, textlow) for pat in comp_words[compartment])
    found_oa = any(re.search(pat, textlow) for pat in patterns)
    # also accept 'degener' when compartment mentioned together with degenerative phrasing
    found_deg = re.search(r'degener', textlow) is not None
    if found_oa:
        # if OA word present and compartment mentioned -> 1
        if found_comp:
            return 1
        # if OA present without compartment but phrase like 'all three compartments' and compartment match
        if re.search(r'all three compartments|three compartments|all compartment', textlow):
            return 1
    if found_deg and found_comp:
        return 1
    return 0


def detect_effusion(text):
    if re.search(r'\b(effusion|joint effusion|derrame|hydrops|hydrops|mayii art[iı]s[ii]|sıvı artışı|sıvı)\b', text, re.I):
        # but negative phrases
        if re.search(r'no (joint )?effusion|no effusion|sin effusion|no significant effusion|no hay derrame|no joint effusion', text, re.I):
            return 0
        return 1
    return 0


def detect_synovitis(text):
    if re.search(r'synovit|synovial hypertrophy|hypertrophy of the synovium|synovial proliferation|synovitis|synoviale', text, re.I):
        return 1
    return 0


def detect_bakers(text):
    if re.search(r'Baker|Baker\'s|Bakerova|popliteal cyst|bakerova', text, re.I):
        return 1
    return 0


def detect_contusion(text):
    if re.search(r'contusion|bone contusion|bone bruise|bone marrow edema|bone marrow oedema|bone marrow edema', text, re.I):
        # if phrase 'no bone contusion' etc
        if re.search(r'no bone contusion|no bone bruise', text, re.I):
            return 0
        return 1
    return 0


def detect_fracture(text):
    if re.search(r'fracture|fractura|frakture|frattura|fractur', text, re.I):
        if re.search(r'no fracture|no evidence of fracture|aucune fracture', text, re.I):
            return 0
        return 1
    return 0

# load mismatches
mismatches = []
with open(mismatches_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        mismatches.append(r)

# load train
with open(train_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    train_rows = list(reader)
    fieldnames = reader.fieldnames

# Map from report snippet to index by searching
changed = []
not_found = []
for m in mismatches:
    excerpt = m.get('report_excerpt') or m.get('report') or ''
    excerpt_short = excerpt.strip().strip('"')[:200]
    matched = None
    for i,r in enumerate(train_rows):
        if excerpt_short and excerpt_short in r['Report']:
            matched = i
            break
        # fallback: match by presence of some words from excerpt
        if len(excerpt_short) > 30 and excerpt_short[:30] in r['Report']:
            matched = i
            break
    if matched is None:
        # try fuzzy: search for first 10 words
        words = excerpt.split()
        key = ' '.join(words[:10])
        for i,r in enumerate(train_rows):
            if key in r['Report']:
                matched = i
                break
    if matched is None:
        not_found.append({'excerpt':excerpt_short})
        continue
    r = train_rows[matched]
    text = r['Report']
    orig = {col:r[col] for col in ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis','Baker\'s','Contusion','Fracture']}
    # detect
    new = {}
    new['ACL'] = str(detect_acl(text))
    new['MCL'] = str(detect_mcl(text))
    new['Medial Meniscus'] = str(detect_medial_meniscus(text))
    new['Lateral Meniscus'] = str(detect_lateral_meniscus(text))
    new['Medial OA'] = str(detect_oa(text,'medial'))
    new['Lateral OA'] = str(detect_oa(text,'lateral'))
    new['PF OA'] = str(detect_oa(text,'pf'))
    new['Effusion'] = str(detect_effusion(text))
    new['Synovitis'] = str(detect_synovitis(text))
    new["Baker's"] = str(detect_bakers(text))
    new['Contusion'] = str(detect_contusion(text))
    new['Fracture'] = str(detect_fracture(text))

    # apply changes
    changed_flag = False
    changes = {}
    for k,v in new.items():
        if r.get(k)!=v:
            changes[k] = {'from':r.get(k),'to':v}
            r[k]=v
            changed_flag = True
    if changed_flag:
        changed.append({'StudyInstanceUID':r['StudyInstanceUID'],'index':matched,'changes':changes,'excerpt':excerpt_short})

# write updated train file
with open(train_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in train_rows:
        writer.writerow(r)

# write log
log_fields = ['StudyInstanceUID','index','excerpt','changed_fields']
with open(log_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=log_fields)
    writer.writeheader()
    for c in changed:
        writer.writerow({'StudyInstanceUID':c['StudyInstanceUID'],'index':c['index'],'excerpt':c['excerpt'],'changed_fields':json.dumps(c['changes'],ensure_ascii=False)})

# final json
final = {
    'total_reviewed': len(mismatches),
    'matched_and_changed': len(changed),
    'not_found': len(not_found),
}
with open(final_json, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2)

print('Done. Reviewed:', final['total_reviewed'], 'Changed:', final['matched_and_changed'], 'Not found:', final['not_found'])
