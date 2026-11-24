"""CPU test: (1) grain-side chat template == HF-map chat template, batch for batch; (2) fast resume (raw-row skip)
reproduces the uninterrupted batch stream exactly."""
import sys, os, json, hashlib, random
sys.path.insert(0, "/home/kevinfarhat/h203-wt/src")
import numpy as np, datasets, transformers
import grain.python as grain
from maxtext.input_pipeline import input_pipeline_utils as u

tok_name = sys.argv[1]
tok = transformers.AutoTokenizer.from_pretrained(tok_name)
random.seed(0)
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
def sent(n): return " ".join(random.choice(words) for _ in range(n))
rows = []
for i in range(400):
    turns = [{"role": "system", "content": "You are helpful."}] if i % 3 == 0 else []
    for t in range(random.randint(1, 3)):
        turns += [{"role": "user", "content": sent(random.randint(3, 40))}, {"role": "assistant", "content": sent(random.randint(3, 120))}]
    rows.append({"messages": turns})
ds_full = datasets.Dataset.from_list(rows).to_iterable_dataset(num_shards=4)
L, B = 512, 2


def make_loader(grain_template, skip_raw=0):
    ds = ds_full.shuffle(seed=0, buffer_size=50)
    col = "messages"
    if not grain_template:
        ds = ds.map(u.apply_chat_template, fn_kwargs={"tokenizer_model": tok, "data_column_name": col})
    os.environ["MAXTEXT_DATA_SKIP_RAW"] = str(skip_raw)
    src = u.HFDataSource(ds, 0, 1, 1, L, [col])
    ops = []
    if grain_template:
        ops.append(u.ApplyChatTemplateTransform(tok, col))
    ops.append(u.SFTPromptMasking(text_column_name=col, completion_only=True, max_target_length=L, unk_id=0))
    ops.append(grain.experimental.PackAndBatchOperation(batch_size=B, length_struct={"inputs": L, "targets": L}))
    sampler = grain.IndexSampler(num_records=len(src), num_epochs=1, shard_options=grain.ShardOptions(0, 1, drop_remainder=False), shuffle=False, seed=0)
    dl = grain.DataLoader(data_source=src, operations=ops, sampler=sampler, worker_count=0,
                          read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=8))
    return dl


def h(b):
    m = hashlib.sha256()
    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x): m.update(k.encode()); walk(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x: walk(y)
        else:
            m.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    walk(b)
    return m.hexdigest()[:12]


N = 12
dl_old = make_loader(False); it = iter(dl_old); old = [h(next(it)) for _ in range(N)]
dl_new = make_loader(True); it = iter(dl_new)
new, raw_at = [], {}
sampler = dl_new._sampler
base, stride = int(sampler[0].index), int(sampler[1].index) - int(sampler[0].index)
for k in range(1, N + 1):
    new.append(h(next(it)))
    st = json.loads(it.get_state())
    seen = st["last_seen_indices"]; idx = int(seen.get(str(st.get("last_worker_index")), max(seen.values())))
    raw_at[k] = (idx - base) // stride
print("old == new (grain template):", old == new)
print("raw_at:", raw_at)
for K in range(1, N):
    dl_r = make_loader(True, skip_raw=raw_at[K]); it = iter(dl_r)
    resumed = [h(next(it)) for _ in range(N - K)]
    print(f"resume after batch {K}: identical to uninterrupted batches {K+1}..{N}:", resumed == new[K:])

# chained: resume after 3, continue 5 batches, take position with offset, resume again
dl_r = make_loader(True, skip_raw=raw_at[3]); it = iter(dl_r)
for j in range(5):
    next(it)
st = json.loads(it.get_state()); seen = st["last_seen_indices"]; idx = int(seen.get(str(st.get("last_worker_index")), max(seen.values())))
pos8 = raw_at[3] + (idx - base) // stride
print("chained position for batch 8:", pos8, "direct:", raw_at[8])
