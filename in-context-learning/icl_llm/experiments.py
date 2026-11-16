"""Part 1b experiments: what does Claude learn from the examples in its prompt?

    E1 regression:  learn a new linear function from k examples; compare with least squares
    E2 labels:      correct vs flipped vs meaningless labels; does it follow the examples or its prior?
    E3 imbalance:   demonstrations skewed 14:2 toward one label; do NEUTRAL reviews follow the majority?

    python -m icl_llm.experiments [regression|labels|imbalance|all] [--model claude-haiku-5-5]

Every call is cached, so re-running replays for free. Results go to runs/part1/<experiment>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .claude import RUNS, ask, spent

# 24 clearly positive and 24 clearly negative one-line reviews, written for this tutorial
POSITIVE = [
    "The soundtrack gave me chills and the ending was perfect.", "I laughed from the first scene to the last.",
    "A warm, clever film that I would happily watch again.", "The acting is superb and the story never drags.",
    "Beautifully shot and genuinely moving.", "This was the best meal I have had all year.",
    "The staff were friendly and the room was spotless.", "Fast delivery and the product works exactly as described.",
    "An inspiring book that I could not put down.", "The concert was electric from start to finish.",
    "Great value, great quality, I will buy again.", "Every chapter was a delight to read.",
    "The kids loved it and so did I.", "Smart writing, lovable characters, a total joy.",
    "The coffee here is outstanding and the bakery is even better.", "A flawless performance by the whole cast.",
    "Comfortable, stylish and it arrived early.", "The tour guide was brilliant and funny.",
    "I was smiling the whole way through.", "Easily the most satisfying game I have played in years.",
    "Crisp sound, long battery life, no complaints at all.", "The garden was stunning and the tea was lovely.",
    "A heartfelt story told with real skill.", "Wonderful service and the dessert was heavenly.",
]
NEGATIVE = [
    "The plot made no sense and I nearly walked out.", "Two hours of my life I will never get back.",
    "The food was cold and the waiter was rude.", "It broke after one day and support never replied.",
    "Dull, predictable and far too long.", "The room smelled of smoke and the bed was broken.",
    "Badly acted and even worse written.", "The worst customer service I have ever experienced.",
    "A boring book that I abandoned halfway.", "The sound was muddy and the band seemed bored.",
    "Overpriced, flimsy and nothing like the photos.", "Every joke fell flat.",
    "The kids were bored within ten minutes.", "Clumsy dialogue and characters I could not care about.",
    "The coffee was burnt and the pastries were stale.", "A lifeless performance from a tired cast.",
    "Uncomfortable, ugly and it arrived late.", "The guide was rude and the tour was a mess.",
    "I was checking my watch the whole time.", "Easily the most frustrating game I have ever played.",
    "Tinny sound and the battery dies in an hour.", "The garden was overgrown and the tea was cold.",
    "A cynical story told without any care.", "Terrible service and the dessert was inedible.",
]
NUMBER_ONLY = "You reply with a single number and nothing else: no words, no working."
# 16 deliberately mixed or neutral reviews: the true label is unclear, so any lean reflects the demonstrations
NEUTRAL = [
    "The film was fine, nothing special either way.", "Some parts were good and some were not.",
    "It arrived on the day they said it would.", "The hotel was about what you would expect for the price.",
    "I finished the book.", "The food was okay, the music was loud.", "It does what it says, more or less.",
    "An average concert on a Tuesday night.", "The second half was different from the first.",
    "I might watch the sequel, I might not.", "The room had a bed, a desk and a window.",
    "It was shorter than I thought it would be.", "Mixed feelings about this one.",
    "The coffee was hot and the cups were small.", "Neither the best nor the worst game this year.",
    "The tour lasted two hours.",
]
LABELSETS = {"correct": ("positive", "negative"), "flipped": ("negative", "positive"), "meaningless": ("foo", "bar")}


def _write(name: str, rows: list[dict]):
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _number(text: str) -> float | None:
    """Strict: the WHOLE reply must be a number. A reply that starts working out loud ("Looking at the data:
    x=-5.53 ...") counts as no answer; taking the first number in it would score an x value as the guess."""
    m = re.fullmatch(r"\s*y?\s*=?\s*(-?\d+(?:\.\d+)?)\s*\.?\s*", text.replace(",", ""))
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------ E1 regression
def regression_task(k: int, trial: int, noise: float = 0.5):
    rng = np.random.default_rng(1000 * k + trial)
    w, b = rng.uniform(-3, 3), rng.uniform(-10, 10)
    x = np.round(rng.uniform(-10, 10, k + 1), 2)
    y = np.round(w * x + b + rng.normal(0, noise, k + 1), 2)
    lines = "\n".join(f"x = {xi:.2f}, y = {yi:.2f}" for xi, yi in zip(x[:k], y[:k]))
    prompt = (f"Here are input/output pairs from an unknown process:\n{lines}\n"
              f"x = {x[k]:.2f}, y = ?\n\nGive your best guess for y, even if you are unsure.")
    return prompt, x, y, w * x[k] + b


def regression(model="claude-haiku-5-5", ks=(1, 2, 4, 8, 16, 32), trials=25):
    jobs = [(k, t) for k in ks for t in range(trials)]

    def one(job):
        k, t = job
        prompt, x, y, truth = regression_task(k, t)
        out = ask(prompt, model=model, max_tokens=30, json_number=True)
        try:
            pred = float(json.loads(out)["y"])
        except Exception:  # noqa: BLE001
            pred = None
        xs, ys = x[:k], y[:k]
        ls = np.polyval(np.polyfit(xs, ys, 1), x[k]) if k >= 2 else ys.mean()
        nn = ys[np.argmin(np.abs(xs - x[k]))]
        return {"model": model, "k": k, "trial": t, "answer": out, "pred": pred, "truth": float(truth),
                "least_squares": float(ls), "nearest_neighbour": float(nn), "mean_of_ys": float(ys.mean())}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    _write(f"regression_{model}", rows)
    return rows


# ------------------------------------------------------------------ E2 labels
def label_prompt(demos: list[tuple[str, int]], query: str, labels: tuple[str, str]) -> str:
    lines = "\n".join(f"Review: {s}\nLabel: {labels[y]}\n" for s, y in demos)
    return (f"{lines}\nReview: {query}\nLabel:\n\nReply with the label only, using one of the labels shown above.")


def labels(model="claude-haiku-5-5", n_demos=16, queries=24):
    data = [(s, 0) for s in POSITIVE] + [(s, 1) for s in NEGATIVE]
    jobs = []
    for cond, labs in LABELSETS.items():
        for q in range(queries):
            rng = np.random.default_rng(q)
            qi = int(rng.integers(len(data)))
            pool_idx = [i for i in range(len(data)) if i != qi]
            pos = [i for i in pool_idx if data[i][1] == 0]
            neg = [i for i in pool_idx if data[i][1] == 1]
            pick = list(rng.choice(pos, n_demos // 2, replace=False)) + list(rng.choice(neg, n_demos // 2, replace=False))
            rng.shuffle(pick)
            jobs.append((cond, labs, q, [data[i] for i in pick], data[qi]))

    def one(job):
        cond, labs, q, demos, (query, y) = job
        out = ask(label_prompt(demos, query, labs), model=model, max_tokens=5).strip().lower()
        followed = out.startswith(labs[y])               # the label the DEMONSTRATIONS imply
        prior = out.startswith(("positive", "negative")[y])  # the true sentiment in words
        return {"model": model, "condition": cond, "query": q, "true_sentiment": "positive" if y == 0 else "negative",
                "answer": out, "follows_demonstrations": followed, "matches_true_sentiment_word": prior}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    _write(f"labels_{model}", rows)
    return rows


# ------------------------------------------------------------------ E3 imbalance
def imbalance(model="claude-haiku-5-5", reps=3):
    """Majority-label bias: demonstrations skewed 14:2 one way or the other, queries are NEUTRAL reviews.
    Measures how often a neutral review is called positive under each skew."""
    data = [(s, 0) for s in POSITIVE] + [(s, 1) for s in NEGATIVE]
    jobs = []
    for skew, (n_pos, n_neg) in {"14 positive : 2 negative": (14, 2), "balanced 8 : 8": (8, 8),
                                 "2 positive : 14 negative": (2, 14)}.items():
        for qi, query in enumerate(NEUTRAL):
            for rep in range(reps):
                rng = np.random.default_rng(1000 * qi + rep)
                pos = [i for i in range(len(data)) if data[i][1] == 0]
                neg = [i for i in range(len(data)) if data[i][1] == 1]
                pick = list(rng.choice(pos, n_pos, replace=False)) + list(rng.choice(neg, n_neg, replace=False))
                rng.shuffle(pick)
                jobs.append((skew, qi, rep, [data[i] for i in pick], query))

    def one(job):
        skew, qi, rep, demos, query = job
        out = ask(label_prompt(demos, query, ("positive", "negative")), model=model, max_tokens=5).strip().lower()
        return {"model": model, "skew": skew, "query": qi, "rep": rep, "answer": out,
                "called_positive": out.startswith("positive"), "called_negative": out.startswith("negative")}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    _write(f"imbalance_{model}", rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["regression", "labels", "imbalance", "all"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    a = ap.parse_args(argv)
    for name, fn in (("regression", regression), ("labels", labels), ("imbalance", imbalance)):
        if a.experiment in (name, "all"):
            fn(model=a.model)
            print(name, "done; total spend so far $%.4f" % spent())


if __name__ == "__main__":
    main()
