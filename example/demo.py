from bert_score import score

with open("mloja1.txt") as f:
    cands = [f.read()]

with open("mloja2.txt") as f:
    refs = [f.read()]

(P, R, F), hashname = score(cands, refs, lang="pt", return_hash=True, model_type="bert-base-multilingual-cased")
print(
    f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
)
