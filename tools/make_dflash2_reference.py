"""Reference drafts of a DFlash2 drafter (tools/train_dflash2.py), for `qwenny dflash2check`: the target in float32 from
Transformers, the drafter in float32 as it was trained.
Usage: python -I tools/make_dflash2_reference.py <model dir or HF id> <drafter dir> <fjordhaus dev.jsonl> <out.safetensors> [gpu] [cpu]
Per prompt i (a ticket with its reply, teacher-forced): ids_i, anchors_i (positions of the anchor tokens), path_i (the selector's
seven drafts after each anchor), first_i (the unary argmax of each block's first draft)."""
import importlib.util, json, sys, tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

spec = importlib.util.spec_from_file_location("train_dflash2", Path(__file__).resolve().parent / "train_dflash2.py")
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)

model_dir, drafter_dir, tests, out = sys.argv[1:5]
gpu, cpu = (sys.argv[5:7] + ["8GiB", "10GiB"][len(sys.argv[5:7]):])
cfg, dc = AutoConfig.from_pretrained(model_dir), json.load(open(Path(drafter_dir) / "config.json"))
fc = dc["dflash_config"]
tok = AutoTokenizer.from_pretrained(model_dir)
offload = tempfile.TemporaryDirectory(prefix="dflash2-ref.", dir=str(Path(out).parent))
target = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="auto", max_memory={0: gpu, "cpu": cpu},
                                              offload_folder=offload.name).eval().requires_grad_(False)
a = type("Args", (), dict(taps=fc["target_layer_ids"], block_size=fc["block_size"], top_k=fc["selector_top_k"], mask_token_id=fc["mask_token_id"],
                          layers=dc["num_hidden_layers"], ff=dc["intermediate_size"], rank=fc["selector_rank"]))
model = td.Drafter(cfg, a).eval().requires_grad_(False)
w = load_file(str(Path(drafter_dir) / "model.safetensors"))
print(model.load_state_dict({k.replace("candidate_selector.", ""): v.float() for k, v in w.items()}))
model = model.cuda()
head, embed = target.get_output_embeddings(), target.model.embed_tokens
table = td.rope_table(cfg.rope_parameters["rope_theta"], 128)

saved = {}
for t in a.taps:
    target.model.layers[t].register_forward_hook(lambda m, i, o, t=t: saved.__setitem__(t, (o[0] if isinstance(o, tuple) else o)[0].float().cuda()))

rows = [json.loads(l) for l in open(tests)]
tensors = {}
for i, r in enumerate(rows[:3]):
    ids = tok.apply_chat_template(r["messages"], enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"][0]
    T = len(ids)
    with torch.no_grad():
        target.model(input_ids=ids[None].cuda(), use_cache=False)
        taps = torch.cat([saved[t] for t in a.taps], -1)
        x_ids = ids.cuda()
        anchors = torch.arange(8, T - 8, max((T - 16) // 48, 1), device="cuda")
        starts = torch.zeros(T, dtype=torch.long, device="cuda")
        cuda_embed = lambda t: embed(t).float().cuda()
        hidden = model(cuda_embed, taps, x_ids, starts, anchors, table, dc["sliding_window"], False)
        paths, firsts = [], []
        for b in range(anchors.shape[0]):
            prev, path = x_ids[anchors[b]], []
            unary = head(hidden[b, 1:]).float().cuda()
            for row in range(a.block_size - 1):
                cands = unary[row].topk(a.top_k).indices
                s = model.select(unary[row][None], hidden[b, row + 1][None], prev[None], cands[None])[0]
                prev = cands[s.argmax()]
                path.append(prev)
            paths.append(torch.stack(path))
            firsts.append(unary[0].argmax())
    tensors |= {f"ids_{i}": ids.int(), f"anchors_{i}": anchors.int().cpu(), f"path_{i}": torch.stack(paths).int().cpu(), f"first_{i}": torch.stack(firsts).int().cpu()}
    print(f"prompt {i}: {T} tokens, {len(paths)} blocks", flush=True)
save_file(tensors, out)
print("wrote", out)
