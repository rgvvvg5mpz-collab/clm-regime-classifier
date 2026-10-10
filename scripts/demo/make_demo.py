#!/usr/bin/env python3
"""Build the narrated CLASP walkthrough video: docs/demo/clasp_demo.mp4 (+ .srt).

    .venv/bin/pip install -r scripts/demo/requirements-demo.txt        # playwright + imageio-ffmpeg (once)
    scripts/run_ui.sh --mock-curation &                                # the UI, with the offline Claude Code stand-in
    .venv/bin/python scripts/demo/make_demo.py                         # ~3 minutes; writes docs/demo/

Three phases:
  1. capture  slides (docs/demo/slides.html) and the live UI at 4K, driven with Playwright through the
              installed Google Chrome; for each shot it records where the spotlight should go.
  2. voice    every narration line spoken separately with macOS `say` (default "Ava (Premium)"),
              so each screen changes exactly when its line starts and the subtitles are exact.
  3. render   every frame in Python (Pillow): eased camera push-ins onto the spotlighted area, an
              animated gold-ringed spotlight, 0.45 s crossfades, chapter lower-thirds in San Francisco;
              4K sources supersampled to 1080p (or --height 2160 for 4K output). Piped to the ffmpeg bundled
              in imageio-ffmpeg: H.264 high profile + AAC + a soft subtitle track.

The recording trains real models and raises real flags; the script backs up app/data/ state first
and restores it afterwards (models, registry, feedback, Tests runs, active model) unless --keep.
Edit the narration in SCENES below and re-run to regenerate the video after UI changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "docs", "demo")
WORK = os.path.join(ROOT, "cache", "demo_build")
FPS = 30
GAP_LINE, GAP_SHOT, GAP_SCENE, LEAD = 0.35, 0.25, 0.75, 0.6   # seconds of silence

# How words are spoken (captions keep the written form).
SPEECH = [
    (r"scripts/run_ui\.sh", "scripts slash run U I"), (r"Qwen3-8B", "Kwen three, eight B"), (r"CLM-8B", "C L M, eight B"),
    (r"FINRA 2210", "Finra twenty-two ten"), (r"FINRA 4530", "Finra forty-five thirty"), (r"FINRA", "Finra"),
    (r"Reg BI", "Reg B I"), (r"macro-F1", "macro F one"), (r"\bF1\b", "F one"), (r"\b0\.98\b", "zero point nine eight"),
    (r"\b0\.8\b", "zero point eight"), (r"\b8710\b", "eighty-seven ten"), (r"\bUI\b", "U I"), (r"System-1", "System One"),
]

# Each scene: shots (what is on screen) and lines [(shot index, narration)].
SCENES = [
    {"id": "intro", "title": "What CLASP is", "shots": [("slide", 1)], "lines": [
        (0, "This is CLASP: Contrastive Labelling And System-1 Prediction."),
        (0, "It's a workbench for building fast text classifiers from a few hundred labelled examples, and for making them better with feedback from the people who use them."),
        (0, "In the next few minutes we'll cover what it is and how it works, and then walk through the whole tool, tab by tab.")]},
    {"id": "system1", "title": "System-1 classifiers", "shots": [("slide", 2)], "lines": [
        (0, "First, what do we mean by a System-1 classifier?"),
        (0, "It's a model that answers one question about a piece of text: which class does it belong to? And it answers by scoring a fixed set of options, rather than writing an answer."),
        (0, "That makes it fast and cheap. A decision takes milliseconds, so it can sit next to a large language model and screen, route or triage everything that model sees and says."),
        (0, "Think of checking every turn of a chat for compliance problems, or sending each support email to the right team.")]},
    {"id": "scoring", "title": "How a text is scored", "shots": [("slide", 3)], "lines": [
        (0, "Here's how CLASP scores a text."),
        (0, "Every class gets a one-sentence description, and the text is compared against each of those descriptions."),
        (0, "Both go through the same encoder, Qwen3-8B, which stays frozen and is never retrained."),
        (0, "On top of it sit two small heads: a state head for the text, and an action head for the class descriptions."),
        (0, "The model measures how similar the two are, turns those similarities into probabilities, and the most likely class wins."),
        (0, "The design comes from CLM-8B, a contrastive language model from Stanford and NVIDIA.")]},
    {"id": "loss", "title": "The contrastive loss", "shots": [("slide", 4)], "lines": [
        (0, "Training uses a contrastive loss over those class descriptions."),
        (0, "For every labelled row, its own class is the positive and every other class is a negative, so you never have to write negative examples by hand."),
        (0, "Only the two heads train: about nineteen million parameters, which takes roughly a minute on a laptop."),
        (0, "So what do you bring? Rows of text and label, a plain sentence describing each class, and one negative class, meaning none of the above, with its own examples."),
        (0, "That last one matters. The model always picks a class, so the negative class is what lets it say that nothing applies.")]},
    {"id": "loop", "title": "The four-tab loop", "shots": [("slide", 5)], "lines": [
        (0, "The tool is organised as a loop of four tabs."),
        (0, "Tab zero, curate: ask Claude Code to research a topic and write a labelled dataset for you."),
        (0, "Tab one, train: pick your data, define the classes, validate it, and train a model."),
        (0, "Tab two, chat and flag: run conversations through the model, and flag every verdict you disagree with."),
        (0, "And tab three, post-train: fold those flags back in, check the new model against the current one, and promote it."),
        (0, "Every model is versioned, so you can always switch back.")]},
    {"id": "example", "title": "The worked example", "shots": [("slide", 6)], "lines": [
        (0, "The repo ships with one complete worked example: a regulatory screener for a broker-dealer's AI chat."),
        (0, "It labels each turn with the rule it might trigger, such as FINRA 2210 for misleading communications, Reg BI for personalised recommendations, or FINRA 4530 for customer complaints. Or it flags nothing."),
        (0, "On its held-out test set, the five-class model scores a macro-F1 of 0.98."),
        (0, "On messages written to look nothing like the training data, it scores about 0.8, and that's the number to plan around."),
        (0, "Its main weakness is reading blunt Reg BI recommendations as FINRA 2210.")]},
    {"id": "start", "title": "Getting started", "shots": [("slide", 7)], "lines": [
        (0, "Getting started takes one command."),
        (0, "Clone the repository and run scripts/run_ui.sh. The first run sets everything up, then serves the app on port 8710."),
        (0, "If you use Claude Code, just open the clone and ask it to run the UI."),
        (0, "You'll need Python, about twenty gigabytes of disk, and either an Apple Silicon Mac or a GPU with twenty-four gigabytes of memory. The demo and training need no API key."),
        (0, "Now let's look at the tool itself.")]},
    {"id": "tour", "subtitle": "Where everything is", "title": "A tour of the workbench", "shots": ["landing", "header", "sidebar"], "lines": [
        (0, "This is the workbench. It opens on the chat tab, with a short welcome."),
        (1, "Across the top are the four steps of the loop, and the model that's currently screening, so you always know which version you're looking at."),
        (2, "The sidebar holds the conversation controls and the screening model selector. It only appears on the tabs that use it."),
        (2, "Hover over any small i icon, and you'll get an explanation of that control.")]},
    {"id": "curate", "subtitle": "Prompt Claude Code to build a labelled dataset", "title": "0 · Data curation", "shots": ["cur_prompt", "cur_feed", "cur_result"], "lines": [
        (0, "Let's start at tab zero, data curation."),
        (0, "You describe the classifier you want in plain language, or start from a template. Here we've chosen review sentiment."),
        (0, "You can set how many rows you want per class, which model Claude Code should use, and a hard budget for the run."),
        (1, "When you press start, Claude Code runs on this machine in its own folder. The activity feed shows each search, each page it reads, and each file it writes."),
        (1, "It can only search the web and read and write files in that folder: no shell commands, and nothing outside it."),
        (2, "When it finishes, CLASP checks what was written: every label has a definition, there's exactly one negative class, and there are enough rows per class."),
        (2, "You can refine the dataset with a follow-up prompt, such as asking for harder examples, or send it straight to the train tab with the classes filled in."),
        (2, "For this recording, Claude Code ran in offline demo mode, so this is a small toy dataset. For the rest of the tour, we'll use the shipped example.")]},
    {"id": "train", "subtitle": "From labelled rows to a scored model in about a minute", "title": "1 · Train", "shots": ["tr_classes", "tr_validate", "tr_progress", "tr_results", "tr_confusion"], "lines": [
        (0, "On the train tab, pick a dataset from the project. Your own files in the data folder appear at the top, followed by anything you've curated, and the examples."),
        (0, "We've loaded the regulatory sample: twenty-one hundred rows across five classes."),
        (0, "Each label gets a display name, the description the model scores against, the speaker who can trigger it, and one is marked as the negative class. Here, they come prefilled."),
        (1, "Validation checks for unknown labels, tiny classes, duplicates and conflicting labels, then splits the data into training, validation and test sets."),
        (2, "Training runs in the background. Because the encoder is frozen, each text is encoded once and cached, and only the heads learn."),
        (3, "A few seconds later, we have results on the held-out test set: accuracy, precision, recall and F1, next to two baselines."),
        (3, "The linear probe is a sanity check: if it beats the trained heads, something's wrong. The zero-shot row shows the untrained heads, close to chance."),
        (4, "The confusion matrix shows exactly where the mistakes are. Rows are the true class, columns are what the model predicted."),
        (4, "One click activates the model, and it becomes the one screening the chat.")]},
    {"id": "chat", "subtitle": "Use the model, and flag what it gets wrong", "title": "2 · Chat & flag", "shots": ["ch_pick", "ch_first", "ch_miss", "ch_details", "ch_flag"], "lines": [
        (0, "On chat and flag, you can replay scripted conversations, so no API key is needed, or chat with a live model."),
        (0, "Each conversation in the list shows how many turns its author expects to be flagged. We'll play the busiest one."),
        (1, "Every turn gets a verdict. Here the user is complaining about a transfer they didn't make, and the model flags FINRA 4530."),
        (1, "Underneath, the transcript's expected label is compared with the model's answer."),
        (2, "Play all runs the rest of the conversation, and not everything is right."),
        (2, "Here the assistant promises the customer they'll get every penny back. That's a promissory statement, so the author expected FINRA 2210, but the model let it through."),
        (3, "Details shows the probability of every class, so you can see how the call was made."),
        (4, "If you disagree with a verdict, flag it. Pick the correct label, add a note if you like, and it's saved as expert data for post-training.")]},
    {"id": "post", "subtitle": "Fold the flags back in, gated and versioned", "title": "3 · Post-train", "shots": ["po_data", "po_options", "po_gate", "po_registry"], "lines": [
        (0, "Tab three, post-train, starts from the selected model."),
        (0, "It lists the flags raised in the chat that fit this model, and you can add a file of corrections from subject-matter experts. We've added the example corrections file."),
        (1, "By default, it continues training the current model gently: a few epochs at a low learning rate, with the expert rows repeated so a handful of corrections can move it, and the original data mixed in so it doesn't forget."),
        (2, "Then comes the gate. The candidate and the current model are scored on the same held-out data, and, for the example, on both out-of-distribution sets."),
        (2, "Any drop in F1, or in recall for a single class, counts as a regression, and promoting past one needs a written reason, which is logged."),
        (2, "Here the candidate improved on both out-of-distribution sets but slipped on one class, so the gate holds it back for a person to decide."),
        (3, "Every model lands in the registry with its version, its parent, its metrics, and its gate result. Our candidate became version two, and any version can be activated again.")]},
    {"id": "classes", "subtitle": "What the selected model can assign", "title": "The classes", "shots": ["classes"], "lines": [
        (0, "Finally, the classes tab draws the selected model's label space: which classes a user turn can get, which an assistant turn can get, and the negative class beneath them."),
        (0, "It's built from the model itself, so it's always right for the version you've selected.")]},
    {"id": "outro", "title": "Where to go next", "shots": [("slide", 8)], "lines": [
        (0, "That's CLASP: curate, train, use, flag, and improve, all in one loop."),
        (0, "The README covers setup and the repository layout, the docs folder has the architecture and a developer guide, and every evaluation run lives in the Tests folder."),
        (0, "Drop your own files into the data folder, and build your own classifier.")]},
]

SPOT_JS_UNUSED = """(el) => { document.querySelectorAll('.demo-spot').forEach(e => e.remove()); if (!el) return;
  const r = el.getBoundingClientRect(); const d = document.createElement('div'); d.className = 'demo-spot';
  Object.assign(d.style, {position: 'fixed', left: (r.left - 8) + 'px', top: (r.top - 8) + 'px', width: (r.width + 16) + 'px',
    height: (r.height + 16) + 'px', border: '4px solid #c8a24d', borderRadius: '14px', zIndex: 99999, pointerEvents: 'none',
    boxShadow: '0 0 0 9999px rgba(11, 42, 74, .34)'}); document.body.append(d); }"""
HIDE_CSS = "#tipbox,#toast{display:none!important} *{caret-color:transparent!important; transition:none!important}"
UI_VIEW = (1440, 810)          # CSS pixels the UI is laid out at
UI_DSF = 8 / 3                 # -> 3840 x 2160 screenshots
SLIDE_DSF = 2                  # slides are 1920 x 1080 CSS -> 3840 x 2160


def api(url: str, path: str, body: dict | None = None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


# ------------------------------------------------------------------ capture
def capture(url: str, shots_dir: str, with_curation: bool) -> None:
    from playwright.sync_api import sync_playwright
    os.makedirs(shots_dir, exist_ok=True)
    slides = "file://" + os.path.join(OUT_DIR, "slides.html")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        rects = {}
        sp = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=SLIDE_DSF)
        for n in range(1, 9):
            sp.goto(f"{slides}?slide={n}&render=1"); sp.wait_for_timeout(300)
            sp.screenshot(path=os.path.join(shots_dir, f"slide{n}.png"))
        ctx = browser.new_context(viewport={"width": UI_VIEW[0], "height": UI_VIEW[1]}, device_scale_factor=UI_DSF)
        pg = ctx.new_page()
        pg.on("dialog", lambda d: d.accept("demo recording"))
        pg.goto(url + "/?demo=1"); pg.add_style_tag(content=HIDE_CSS)
        pg.wait_for_function("document.getElementById('clfStatus').textContent.startsWith('Model ready')", timeout=180000)
        pg.wait_for_timeout(800)

        def shot(name, target=None, settle=350):
            rect = None
            if target is not None:
                loc = pg.locator(target).first if isinstance(target, str) else target
                loc.scroll_into_view_if_needed(); pg.wait_for_timeout(250)
                bb = loc.bounding_box()
                if bb:   # clip to the viewport, in screenshot pixels
                    x0, y0 = max(0, bb["x"]), max(0, bb["y"])
                    x1, y1 = min(UI_VIEW[0], bb["x"] + bb["width"]), min(UI_VIEW[1], bb["y"] + bb["height"])
                    rect = [round(v * UI_DSF) for v in (x0, y0, x1, y1)]
            pg.wait_for_timeout(settle)
            pg.screenshot(path=os.path.join(shots_dir, f"{name}.png"))
            rects[name] = rect

        def tab(view):
            pg.click(f'.tab[data-view="{view}"]'); pg.wait_for_timeout(700)

        # tour
        shot("landing"); shot("header", ".topbar"); shot("sidebar", "#sidebar")
        # 0 curation
        if with_curation:
            tab("curateView")
            pg.wait_for_function("document.getElementById('curStatus').innerText.includes('ready')", timeout=30000)
            pg.select_option("#curTemplate", "sentiment"); pg.fill("#curRows", "20")
            shot("cur_prompt", "#c-step1")
            old = pg.evaluate("() => document.getElementById('curFeed').innerText")   # the server may still hold an earlier job
            pg.click("#curStart")
            pg.wait_for_function("(old) => { const f = document.getElementById('curFeed').innerText; return f !== old && f.startsWith('Finished') "
                                 "&& document.getElementById('curResult').innerText.length > 80; }", arg=old, timeout=180000)
            pg.wait_for_timeout(600)
            shot("cur_feed", "#c-step2"); shot("cur_result", "#c-step3")
        # 1 train
        tab("trainView")
        pg.select_option("#trainExample", "examples/regulatory/samples/regime_5way_sample.csv")
        pg.wait_for_function("document.querySelectorAll('#classSpec tr[data-label]').length > 0 && document.querySelector('#classSpec textarea').value.length > 0", timeout=30000)
        pg.fill("#trainName", "regime_demo")
        shot("tr_classes", "#t-step2")
        pg.click("#trainValidate"); pg.wait_for_function("!document.getElementById('t-step4').hidden", timeout=60000)
        shot("tr_validate", "#t-step3")
        pg.click("#trainStart")
        pg.wait_for_function("/epoch \\d+\\/\\d+/.test(document.getElementById('trainJob').innerText)", timeout=120000)
        shot("tr_progress", "#t-step4", settle=150)
        pg.wait_for_function("!document.getElementById('t-step5').hidden && document.getElementById('trainResults').innerText.length > 50", timeout=300000)
        pg.wait_for_timeout(600)
        shot("tr_results", pg.locator("#trainResults table").first)
        shot("tr_confusion", "#trainResults table.cm")
        pg.locator("#trainResults button").filter(has_text=re.compile("Activate|Promote")).first.click()
        pg.wait_for_timeout(2500)
        # 2 chat & flag
        tab("chatView")
        pg.evaluate("() => { const s = document.getElementById('conv'); s.value = '13'; s.dispatchEvent(new Event('change')); }")
        pg.wait_for_timeout(500)
        shot("ch_pick", "#transcriptPanel")
        pg.click("#playNext"); pg.wait_for_function("document.querySelectorAll('.msg .chip').length >= 2", timeout=60000)
        shot("ch_first", pg.locator(".msg.user").first)
        pg.click("#playAll"); pg.wait_for_function("document.getElementById('playStatus').textContent === 'End of conversation'", timeout=120000)
        miss = pg.locator(".msg.assistant", has=pg.locator(".expected.miss")).first
        if miss.count() == 0:
            miss = pg.locator(".msg", has=pg.locator(".expected.miss")).first
        shot("ch_miss", miss)
        miss.locator(".linkbtn", has_text="details").click(); pg.wait_for_timeout(300)
        shot("ch_details", miss)
        miss.locator(".linkbtn", has_text=re.compile("Flag")).click(); pg.wait_for_timeout(300)
        pg.evaluate("""(m) => { const f = m.querySelector('.flagform'); const exp = (m.querySelector('.expected')||{}).textContent || '';
            const map = {'Reg BI':'reg_bi','FINRA 2210':'finra_2210','FINRA 4530':'finra_4530','SEC 17a-3/4':'sec_17a3_17a4','nothing':'no_flag'};
            const k = Object.keys(map).find((x) => exp.startsWith('expected ' + x)); const sel = f.querySelector('select');
            if (k && [...sel.options].some((o) => o.value === map[k])) sel.value = map[k]; f.querySelector('[name=note]').value = 'promising every penny back is a promissory statement'; }""",
                    miss.element_handle())
        shot("ch_flag", miss)   # the form, filled in
        miss.locator(".flagform button[type=submit]").click(); pg.wait_for_timeout(600)
        # flag the remaining misses too, so post-training has a few rows
        pg.evaluate("""async () => { for (const m of document.querySelectorAll('.msg')) { const e = m.querySelector('.expected.miss'); const b = [...m.querySelectorAll('.linkbtn')].find((x) => /^Flag/.test(x.textContent));
            if (!e || !b) continue; b.click(); await new Promise((r) => setTimeout(r, 150)); const f = m.querySelector('.flagform'); if (!f) continue;
            f.requestSubmit(); await new Promise((r) => setTimeout(r, 300)); } }""")
        # 3 post-train
        tab("postView")
        pg.wait_for_function("document.getElementById('postFeedback').innerText.includes('expert flags')", timeout=30000)
        pg.select_option("#postExample", "examples/regulatory/samples/regime_5way_expert_corrections.csv")
        pg.wait_for_function("document.getElementById('postUploadInfo').innerText.includes('rows ready')", timeout=60000)
        pg.fill("#postName", "regime_demo")
        shot("po_data", pg.locator("#postView .step").nth(1))
        shot("po_options", pg.locator("#postView .step").nth(2))
        pg.click("#postStart")
        pg.wait_for_function("!document.getElementById('p-results').hidden && document.getElementById('postResults').innerText.length > 50", timeout=600000)
        pg.wait_for_timeout(800)
        gate = pg.locator("#postResults table").filter(has_text="Active macro-F1").first
        shot("po_gate", gate if gate.count() else "#p-results")
        shot("po_registry", pg.locator("#postView .step").last)
        # classes
        tab("aboutView"); shot("classes", "#classDiagram")
        browser.close()
        json.dump(rects, open(os.path.join(shots_dir, "rects.json"), "w"), indent=1)


# ------------------------------------------------------------------ voice + assembly
def speakable(text: str) -> str:
    for pat, rep in SPEECH:
        text = re.sub(pat, rep, text)
    return text


def say(text: str, voice: str, rate: int, path: str) -> float:
    subprocess.run(["say", "-v", voice, "-r", str(rate), "--file-format=WAVE", "--data-format=LEI16@24000",
                    "-o", path, speakable(text)], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def srt_time(t: float) -> str:
    h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def assemble(shots_dir: str, voice: str, rate: int, out_mp4: str, with_curation: bool, height: int = 1080) -> dict:
    os.makedirs(os.path.join(WORK, "audio"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "segments"), exist_ok=True)
    scenes = [s for s in SCENES if with_curation or s["id"] != "curate"]
    rects = json.load(open(os.path.join(shots_dir, "rects.json"))) if os.path.exists(os.path.join(shots_dir, "rects.json")) else {}
    sr = 24000
    pcm = bytearray(b"\x00\x00" * int(LEAD * sr))
    t = LEAD
    cues, segments, chapters = [], [], []
    CARD = 1.5   # seconds of silent chapter card before each UI chapter
    for si, sc in enumerate(scenes):
        is_ui = not isinstance(sc["shots"][0], tuple)
        chapters.append((t if si else 0.0, sc["title"]))
        if is_ui:
            frames = round(CARD * FPS)
            pcm += b"\x00\x00" * int(round(frames / FPS * sr)); t += frames / FPS
            segments.append({"card": sc["title"], "frames": frames, "subtitle": sc.get("subtitle", "")})
        for k, shot in enumerate(sc["shots"]):
            name = f"slide{shot[1]}" if isinstance(shot, tuple) else shot
            start = t
            lines = [txt for (i, txt) in sc["lines"] if i == k]
            for li, txt in enumerate(lines):
                wav = os.path.join(WORK, "audio", f"{sc['id']}_{k}_{li}.wav")
                d = say(txt, voice, rate, wav)
                with wave.open(wav) as w:
                    pcm += w.readframes(w.getnframes())
                cues.append((t, t + d, txt)); t += d
                gap = GAP_LINE if li < len(lines) - 1 else (GAP_SHOT if k < len(sc["shots"]) - 1 else GAP_SCENE)
                pcm += b"\x00\x00" * int(gap * sr); t += gap
            dur = t - start
            if si == 0 and k == 0:
                dur += LEAD; start -= LEAD
            frames = math.ceil(dur * FPS); pad = frames / FPS - dur   # keep audio and video frame-aligned
            pcm += b"\x00\x00" * int(round(pad * sr)); t += pad
            is_slide = isinstance(shot, tuple)
            segments.append({"png": os.path.join(shots_dir, f"{name}.png"), "frames": frames, "rect": rects.get(name),
                             "kind": "slide" if is_slide else "ui", "new_scene": k == 0,
                             "title": None})
    wav_all = os.path.join(WORK, "narration.wav")
    with wave.open(wav_all, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(bytes(pcm))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render import Renderer
    ff = ffmpeg()
    video = os.path.join(WORK, "video.mp4")
    r = Renderer(ff, video, width=round(height * 16 / 9), height=height, fps=FPS)
    t0 = time.time()
    for n, seg in enumerate(segments):
        if "card" in seg:
            r.card(seg["card"], seg["frames"], subtitle=seg.get("subtitle", ""), first=n == 0)
        else:
            r.shot(**seg, first=n == 0, last=n == len(segments) - 1)
        print(f"\r  rendering shot {n + 1}/{len(segments)}  ({r.frames} frames, {time.time() - t0:.0f}s)", end="", flush=True)
    r.close(); print()
    srt = os.path.splitext(out_mp4)[0] + ".srt"
    with open(srt, "w") as f:
        for n, (a, b, txt) in enumerate(cues, 1):
            f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-i", wav_all, "-i", srt, "-map", "0:v", "-map", "1:a", "-map", "2:s",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                    "-metadata", "title=CLASP walkthrough", "-movflags", "+faststart", "-shortest", out_mp4], check=True)
    return {"duration_s": round(t, 1), "chapters": chapters, "lines": len(cues), "srt": srt}


# ------------------------------------------------------------------ state backup / restore
def snapshot(url: str) -> dict:
    files = {}
    for f in ("models.json", "feedback.jsonl", "turns.jsonl", "promotions.jsonl"):
        p = os.path.join(ROOT, "app", "data", f)
        files[f] = open(p, "rb").read() if os.path.exists(p) else None
    return {"files": files, "models": set(glob.glob(os.path.join(ROOT, "models", "*"))),
            "tests": set(glob.glob(os.path.join(ROOT, "Tests", "20*"))), "uploads": set(glob.glob(os.path.join(ROOT, "app", "data", "uploads", "*"))),
            "active": api(url, "/api/config")["active"]}


def restore(url: str, snap: dict) -> None:
    try:
        api(url, "/api/train/activate", {"checkpoint": snap["active"], "reason": "demo recording: restore"})
    except Exception as e:
        print("could not re-activate", snap["active"], e)
    for f, data in snap["files"].items():
        p = os.path.join(ROOT, "app", "data", f)
        if data is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            open(p, "wb").write(data)
    for p in set(glob.glob(os.path.join(ROOT, "models", "*"))) - snap["models"]:
        os.remove(p)
    for p in set(glob.glob(os.path.join(ROOT, "Tests", "20*"))) - snap["tests"]:
        shutil.rmtree(p)
    for p in set(glob.glob(os.path.join(ROOT, "app", "data", "uploads", "*"))) - snap["uploads"]:
        shutil.rmtree(p)
    shutil.rmtree(os.path.join(ROOT, "app", "data", "curation"), ignore_errors=True) if not snap.get("keep_curation") else None
    subprocess.run([sys.executable, os.path.join(ROOT, "Tests", "make_reports.py")], check=False, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8710")
    ap.add_argument("--voice", default="Ava (Premium)")
    ap.add_argument("--rate", type=int, default=178, help="words per minute for `say`")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "clasp_demo.mp4"))
    ap.add_argument("--skip-curation", action="store_true", help="omit tab 0 (e.g. Claude Code not available)")
    ap.add_argument("--reuse-shots", action="store_true", help="skip capture; re-voice and re-assemble existing shots")
    ap.add_argument("--keep", action="store_true", help="keep the models / flags / runs the recording creates")
    ap.add_argument("--height", type=int, default=1080, choices=[1080, 1440, 2160], help="output height (16:9)")
    a = ap.parse_args()
    voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    if a.voice not in voices:
        sys.exit(f'voice "{a.voice}" is not installed; add it in System Settings > Accessibility > Spoken Content > System voice > Manage voices')
    shots_dir = os.path.join(WORK, "shots")
    with_curation = not a.skip_curation
    if not a.reuse_shots:
        h = api(a.url, "/api/health")
        if h.get("state") != "ready":
            sys.exit(f"the UI at {a.url} is not ready ({h.get('state')}); start it and wait for the model to load")
        if with_curation and not api(a.url, "/api/curate/status").get("logged_in"):
            sys.exit("Claude Code is not usable by the server; start it with scripts/run_ui.sh --mock-curation, or pass --skip-curation")
        snap = snapshot(a.url)
        t0 = time.time()
        try:
            capture(a.url, shots_dir, with_curation)
        finally:
            if not a.keep:
                restore(a.url, snap)
        print(f"captured {len(os.listdir(shots_dir))} frames in {time.time() - t0:.0f}s")
    info = assemble(shots_dir, a.voice, a.rate, a.out, with_curation, height=a.height)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({**info, "voice": a.voice, "rate": a.rate, "video": os.path.relpath(a.out, ROOT)},
              open(os.path.join(OUT_DIR, "build_info.json"), "w"), indent=2)
    m, s = divmod(info["duration_s"], 60)
    print(f"wrote {os.path.relpath(a.out, ROOT)}: {int(m)}:{s:04.1f}, {info['lines']} narration lines, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    for ts, title in info["chapters"]:
        print(f"  {int(ts // 60)}:{int(ts % 60):02d}  {title}")


if __name__ == "__main__":
    main()
