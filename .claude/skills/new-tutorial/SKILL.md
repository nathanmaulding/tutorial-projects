---
name: new-tutorial
description: Start a new tutorial in this repo from _template/: create the folder, notebook, README and requirements, write the explanation-plus-code notebook, run it to save outputs, and add it to the top-level README table. Use when asked to add, create or start a tutorial or notebook here.
---

# New tutorial

1. **Agree the scope with the user first:** the topic, the 4 to 7 concepts in teaching order, the target runtime (aim for under 10 minutes on free Colab CPU), and any dependencies beyond numpy and matplotlib. Prefer runs that need no GPU and no API key. If a section needs a key, make it optional and skip it cleanly when the key is missing.
2. **Create the folder:** `<kebab-case-topic>/`, copying `_template/README.md`, `_template/requirements.txt` and `_template/tutorial.ipynb` (renamed to `<snake_case_topic>.ipynb`).
3. **Write the notebook in the house style**, following `gradient-descent/` as the reference:
   - an opening cell with a hook, "What you'll learn", the runtime and the Colab badge
   - a setup cell with fixed seeds
   - for each concept: an explanation (markdown, with an equation or picture where it helps), then code, then a "**What just happened**" note
   - at least two "⏸️ **Pause and predict**" prompts, since these work well on video
   - a sanity check against a known answer wherever one exists
   - a Recap table, then a "Try it yourself" exercise list
   - code short and readable on screen: small functions, no hidden helper modules
4. **Run it top to bottom** with `jupyter nbconvert --to notebook --execute --inplace <notebook>`, so outputs are saved. Read the printed outputs and look at every figure, and confirm the text's claims match what the code produced.
5. **Fill in the tutorial README**, and add a row to the table in the top-level `README.md`.
6. **Don't commit or push** unless the user asks.
