"""Original protected language fixtures, forward-only generations and blinded scoring.

The five tasks share 20 scenarios per split. Report 20 contexts, not 100
independent subject areas. These closed-context tasks complement the historical
open-ended grammar, vocabulary, factual and paragraph benchmarks.
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest

VERSION = "codexa-language-eval-1"
TASKS = ("reading", "instructions", "explanation", "summarization", "dialogue")
DIMENSIONS = ("grammar", "relevance", "coherence", "completeness", "task_success")
SETTINGS: dict[str, Any] = {
    "max_tokens": 256,
    "seed": 42,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "precision": "bf16",
}
RUBRIC = {
    "scale": {"0": "failed", "1": "partly satisfied", "2": "fully satisfied"},
    "grammar": "Assess comprehensible English sentence construction across the whole response.",
    "relevance": "Assess whether the response addresses this prompt without unrelated content.",
    "coherence": "Assess logical connections and consistency; repetitive loops cannot score 2.",
    "completeness": "Assess whether the requested answer is complete; flag truncation separately.",
    "task_success": "Assess required facts, constraints and absence of contradiction against the passage.",
    "uncertain": "Record an unresolved judgment explicitly; it is not a passing judgment.",
    "reference": "References are factual guides, not exact-match answers. Accept faithful paraphrases.",
    "independence": "Two distinct reviewers score independently before checkpoint mapping is revealed.",
}

# Each entry is original fictional microprose, question, answer, causal question,
# and causal reference. No external corpus or generated model output is used.
SCENARIOS: dict[str, list[tuple[str, str, str, str, str]]] = {
    "dev": [
        (
            "A library's return box filled during a holiday. Staff placed a second box beside it before opening. Readers could then return books without waiting for space.",
            "What did staff add?",
            "A second return box.",
            "Why did adding another box help readers?",
            "It provided room for returns when the first box was full.",
        ),
        (
            "A school garden's seedlings dried out during windy afternoons. The class covered the soil with leaves. The soil stayed damp for longer, so the class watered less often.",
            "What covered the soil?",
            "Leaves covered the soil.",
            "Why did the class water less often?",
            "The leaf cover helped the soil stay damp for longer.",
        ),
        (
            "A repair cafe accepted broken lamps but had no spare plugs. Volunteers borrowed a box of plugs from a nearby workshop. They could then finish repairs that had been waiting.",
            "Where did volunteers get spare plugs?",
            "They borrowed plugs from a nearby workshop.",
            "Why had some repairs been waiting?",
            "The cafe lacked spare plugs needed to finish them.",
        ),
        (
            "A museum's small labels were difficult to read from a wheelchair. Staff moved the labels lower and printed larger letters. Visitors could read them without asking someone to lift a card.",
            "What two label changes did staff make?",
            "They moved labels lower and made the letters larger.",
            "Why did the new labels improve access?",
            "Their position and larger letters made them easier to read from a wheelchair.",
        ),
        (
            "A village bus usually stopped outside the market. Road repairs closed that street for a week. A notice directed passengers to a temporary stop beside the clinic.",
            "Where was the temporary bus stop?",
            "Beside the clinic.",
            "Why was the bus stop moved?",
            "Road repairs closed the street outside the market.",
        ),
        (
            "A choir could not hear its quietest singers over a noisy fan. The director switched off the fan during practice and opened a window between songs. The singers could hear each other more clearly.",
            "When was the window opened?",
            "Between songs.",
            "Why did switching off the fan help the choir?",
            "Removing its noise made the quiet singers easier to hear.",
        ),
        (
            "A bakery received many orders for a festival. Its oven could bake only one tray at a time. The baker scheduled earlier batches rather than promising more bread than the oven could produce.",
            "What limited bread production?",
            "The oven could bake only one tray at a time.",
            "Why did the baker schedule earlier batches?",
            "Earlier batches provided more baking time despite the oven's limited capacity.",
        ),
        (
            "A walking group found a muddy section of its usual path. The leader chose a longer route on firm ground. The group arrived later but avoided the slippery slope.",
            "What did the group avoid?",
            "The slippery slope.",
            "Why did the group arrive later?",
            "It took a longer route to avoid muddy ground.",
        ),
        (
            "A classroom recorder missed soft voices at the back. Moving it to the centre captured those voices but also recorded chair noise. The teacher asked pupils to keep chairs still while speaking.",
            "What new noise did the recorder capture?",
            "Chair noise.",
            "Why did the teacher ask pupils to keep chairs still?",
            "Chair noise was being recorded along with their voices.",
        ),
        (
            "A shop sold fruit in mixed boxes. Some customers wanted only apples, so the owner added an apple-only box while keeping the mixed option. Customers could choose either kind.",
            "Which original option remained available?",
            "The mixed fruit box.",
            "Why did the owner add another box option?",
            "Some customers wanted apples rather than mixed fruit.",
        ),
        (
            "A local radio station lost its internet connection before an interview. The host played a saved recording instead. The live interview was moved to the next morning.",
            "When was the live interview moved to?",
            "The next morning.",
            "Why did the host play a saved recording?",
            "The station lost its internet connection before the live interview.",
        ),
        (
            "A costume maker used a delicate fabric for a stage cape. A test seam tore when pulled. She added a stronger lining before the actor wore the cape on stage.",
            "What did the maker add?",
            "A stronger lining.",
            "Why was a lining added before the performance?",
            "A test seam tore, showing that the delicate fabric needed reinforcement.",
        ),
        (
            "A shared kitchen had identical jars of salt and sugar. A cook confused them while making soup. The team labelled each jar and stored them on different shelves.",
            "How did the team distinguish the jars?",
            "They labelled the jars and put them on different shelves.",
            "Why did the team change the storage arrangement?",
            "The identical jars had caused a cook to confuse salt and sugar.",
        ),
        (
            "A rescue shelter had a quiet room for nervous animals. Building work made the neighbouring room loud. Staff moved the quiet area to the far end of the building until the work ended.",
            "How long was the quiet area moved for?",
            "Until the building work ended.",
            "Why did staff relocate the quiet area?",
            "Building noise nearby was unsuitable for nervous animals.",
        ),
        (
            "A newspaper received a photograph with no photographer's name. The editor delayed publication and contacted the sender. The photo appeared only after its credit and permission were confirmed.",
            "When did the photo appear?",
            "After its credit and permission were confirmed.",
            "Why did the editor delay publication?",
            "The photographer's identity and permission had not yet been confirmed.",
        ),
        (
            "A sports club stored wet uniforms in a closed bag. They smelled unpleasant the next day. The coach arranged a drying rack so uniforms could dry before being packed.",
            "What storage change did the coach arrange?",
            "A drying rack before packing.",
            "Why did drying uniforms before packing address the problem?",
            "The problem followed storing uniforms while wet in a closed bag.",
        ),
        (
            "A community centre offered an evening class upstairs. A broken lift prevented some learners from attending. The organiser moved the class downstairs while the lift was repaired.",
            "Where was the class moved?",
            "Downstairs.",
            "Why did moving the class help attendance?",
            "It removed the need to use the broken lift to reach the class.",
        ),
        (
            "A potter tested a new glaze on a small tile. It changed colour during firing, unlike the unfired sample. She waited for the fired test before glazing a large order.",
            "Which sample guided the final decision?",
            "The fired test tile.",
            "Why did the potter wait before glazing the order?",
            "The glaze changed colour in firing, so the unfired sample was misleading.",
        ),
        (
            "A book club's meeting room was booked by another group. Members agreed to meet in the courtyard if it stayed dry. Rain began at noon, so they used a spare classroom instead.",
            "Where did the club actually meet?",
            "In a spare classroom.",
            "Why was the courtyard plan abandoned?",
            "Rain began, and the courtyard plan depended on dry weather.",
        ),
        (
            "A neighbourhood tool library had one working drill. Two borrowers requested it on the same day. The volunteer arranged separate morning and afternoon slots instead of accepting overlapping bookings.",
            "How were both requests accommodated?",
            "With separate morning and afternoon borrowing slots.",
            "Why did the bookings need different times?",
            "Only one working drill was available.",
        ),
    ],
    "test": [
        (
            "A harbour ferry's normal landing was damaged in a storm. The crew used a sheltered jetty farther along the shore. A sign at the old landing explained where passengers should go.",
            "Where did passengers board instead?",
            "At a sheltered jetty farther along the shore.",
            "Why was a sign placed at the old landing?",
            "It directed passengers to the replacement boarding point after storm damage.",
        ),
        (
            "A seed exchange stored packets in an open shed. Rain blew inside and damaged the paper labels. Volunteers moved the packets indoors and copied readable labels onto waterproof tags.",
            "What happened to the paper labels?",
            "Rain damaged them.",
            "Why did volunteers use waterproof tags?",
            "Rain had damaged paper labels, so tags better protected the identifying information.",
        ),
        (
            "An amateur theatre sold tickets for a room with narrow doorways. A bulky prop could not fit through them. The designer rebuilt it in sections that could be assembled inside.",
            "How was the prop changed?",
            "It was rebuilt in sections for assembly inside.",
            "Why did the designer divide the prop into sections?",
            "The complete prop was too bulky to pass through the narrow doorways.",
        ),
        (
            "A clinic's telephone message gave an old opening time. Patients arrived before the doors opened. The receptionist corrected the recording and placed the current hours by the entrance.",
            "What information was corrected?",
            "The opening time on the telephone recording.",
            "Why had patients arrived too early?",
            "The telephone message still gave an old opening time.",
        ),
        (
            "A mapmaker marked a footbridge as open. A ranger reported that floods had swept it away. The next map showed the crossing as closed and suggested the road bridge upstream.",
            "Which crossing did the revised map suggest?",
            "The road bridge upstream.",
            "Why did the mapmaker change the map?",
            "The ranger reported that flooding had destroyed the footbridge.",
        ),
        (
            "A rooftop weather sensor was shaded by a growing vine. Its readings no longer matched the nearby open-air sensor. The caretaker trimmed the vine before checking the instruments again.",
            "What shaded the rooftop sensor?",
            "A growing vine.",
            "Why was the vine trimmed before the next comparison?",
            "Its shade could affect the rooftop sensor's readings.",
        ),
        (
            "A sign-language workshop originally faced a bright window. The glare made hands and faces hard to see. The tutor turned the seating so light fell from the side.",
            "How was the seating changed?",
            "It was turned so light came from the side.",
            "Why did changing the seating improve visibility?",
            "It avoided facing the bright window and reduced glare.",
        ),
        (
            "A bicycle courier planned a shortcut through a park. A notice said the gates would close before the delivery time. The courier chose public roads instead and informed the customer of the longer route.",
            "Which route did the courier choose?",
            "Public roads.",
            "Why was the park shortcut unsuitable?",
            "The park gates would close before the delivery time.",
        ),
        (
            "A science club's demonstration required a dark room. The curtains left a bright gap, so the projected image was faint. Members covered the gap before inviting the audience back.",
            "What made the projected image faint?",
            "Light through a gap in the curtains.",
            "Why did covering the gap help the demonstration?",
            "It darkened the room, making the projected image easier to see.",
        ),
        (
            "A food cooperative received a delivery with one unlabelled container. The coordinator kept it separate rather than guessing its contents. It was distributed only after the supplier identified it.",
            "When was the container distributed?",
            "After the supplier identified it.",
            "Why was the unlabelled container kept separate?",
            "Its contents were unknown and the coordinator would not guess.",
        ),
        (
            "A storyteller used an old recording at a school event. Several words were obscured by static. She paused the recording and explained only the parts she could verify from a written transcript.",
            "What did the storyteller use to verify words?",
            "A written transcript.",
            "Why did she limit her explanation to verified parts?",
            "Static obscured some recorded words, so their meaning was uncertain.",
        ),
        (
            "A warehouse stacked fragile cups beneath heavy boxes. Several cups cracked during transport. The packing team moved the cups above the heavy items and added a rigid divider.",
            "Where were fragile cups placed after the change?",
            "Above the heavy items.",
            "Why did the packing arrangement change?",
            "Cups had cracked when heavy boxes were stacked above them.",
        ),
        (
            "A youth centre's online form required an email address. Some families had no email access. Staff added a paper registration option without removing the online form.",
            "Which registration method remained available?",
            "The online form.",
            "Why was a paper option added?",
            "Some families could not use an email-based registration process.",
        ),
        (
            "A travelling exhibition used numbered crates. Two labels came loose, leaving their destinations unclear. The organiser checked the packing list before sending those crates to any gallery.",
            "What did the organiser check?",
            "The packing list.",
            "Why were the two crates not sent immediately?",
            "Their destination labels had come loose, so their destinations were unclear.",
        ),
        (
            "A birdwatching group planned to enter a reserve at dawn. The warden announced a temporary nesting-area closure. The group watched from the public hide outside the closed area instead.",
            "Where did the group watch birds?",
            "From the public hide outside the closed area.",
            "Why did the group change its viewing location?",
            "The nesting area was temporarily closed.",
        ),
        (
            "A brass band rehearsed beside an open study room. Students complained that they could not concentrate. The centre moved rehearsal to a room across the courtyard and kept its doors closed.",
            "Where was rehearsal moved?",
            "To a room across the courtyard.",
            "Why did the centre move the rehearsal?",
            "Band noise was disturbing students in the study room.",
        ),
        (
            "A small publisher found a missing paragraph in a proof copy. Printing had not begun. The editor restored the paragraph and checked the surrounding page numbers before approving the book.",
            "Had printing begun when the omission was found?",
            "No, printing had not begun.",
            "Why were the surrounding page numbers checked?",
            "Restoring missing text could affect the surrounding pages and their numbering.",
        ),
        (
            "A hiking hut relied on a spring for drinking water. A fallen branch broke the supply pipe. The caretaker brought sealed water containers and closed the tap until the pipe was repaired.",
            "What temporary water supply was provided?",
            "Sealed water containers.",
            "Why was the tap closed?",
            "The supply pipe was broken and needed repair.",
        ),
        (
            "A dance instructor recorded a routine facing the class. Learners copying the video confused left and right. She made a second recording from behind, with spoken direction cues.",
            "From which viewpoint was the second recording made?",
            "From behind the instructor.",
            "Why was another recording useful?",
            "The first viewpoint caused left-right confusion; the new viewpoint and spoken cues addressed it.",
        ),
        (
            "A town archive received letters in no clear order. Their envelopes still carried dates. The archivist used those dates to arrange the letters and marked undated ones separately rather than inventing dates.",
            "How were undated letters handled?",
            "They were marked separately without invented dates.",
            "Why did the archivist use envelope dates?",
            "The letters arrived out of order and the envelope dates provided evidence for ordering them.",
        ),
    ],
}


def fixture_rows() -> list[dict[str, Any]]:
    rows = []
    for split, scenarios in SCENARIOS.items():
        for i, (passage, question, answer, why, reason) in enumerate(scenarios):
            prompts = {
                "reading": f"Read the passage and answer the question.\nPassage: {passage}\nQuestion: {question}\nAnswer:",
                "instructions": f"Use the passage below. Return exactly two bullet points, starting with '- '. The first must answer the question; the second must explain the reason. Do not add an introduction.\nPassage: {passage}\nQuestion: {question}\nReason question: {why}\nResponse:",
                "explanation": f"Explain the following situation in two or three connected sentences, using only the passage.\nPassage: {passage}\nQuestion: {why}\nExplanation:",
                "summarization": f"Summarize this passage in one or two sentences. Include the problem and the response to it. Do not add new events.\nPassage: {passage}\nSummary:",
                "dialogue": f"User: Let's discuss this passage: {passage}\nAssistant: I have read the passage.\nUser: {question} Please explain why that happened, using the passage we just discussed.\nAssistant:",
            }
            for task in TASKS:
                required = [answer, reason] if task != "reading" else [answer]
                if task == "explanation":
                    required = [reason]
                rows.append(
                    {
                        "id": f"language-v1-{split}-{i:02d}-{task}",
                        "split": split,
                        "task": task,
                        "family": f"scenario-{split}-{i:02d}",
                        "prompt": prompts[task],
                        "passage": passage,
                        "question": question,
                        "reference": passage if task == "summarization" else " ".join(required),
                        "required_facts": required if task != "summarization" else [passage],
                        "rubric": dict.fromkeys(DIMENSIONS, "See frozen rubric.json"),
                        "license": "CC0-1.0",
                    }
                )
    return rows


def freeze(destination: Path) -> dict[str, Any]:
    """Create a new fixture directory; refuse even matching existing destinations."""
    destination.mkdir(parents=True, exist_ok=False)
    rows = fixture_rows()
    files = {}
    for split in ("dev", "test"):
        path = destination / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["split"] == split)
        )
        files[path.name] = digest(path)
    atomic(destination / "rubric.json", RUBRIC)
    files["rubric.json"] = digest(destination / "rubric.json")
    manifest = {
        "version": VERSION,
        "license": "CC0-1.0 original fictional fixtures",
        "counts": {
            split: dict(Counter(r["task"] for r in rows if r["split"] == split))
            for split in ("dev", "test")
        },
        "files": files,
        "settings": SETTINGS,
        "contexts_per_split": 20,
        "limitations": "Five tasks share each scenario; 100 cases per split are 20 correlated context groups. Closed-context tasks do not establish broad factual knowledge or real user dialogue ability.",
        "source_sha256": digest(Path(__file__)),
        "readiness": "unreviewed; no competence claim",
    }
    atomic(destination / "manifest.json", manifest)
    return manifest


def load_fixtures(destination: Path) -> list[dict[str, Any]]:
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("settings") != SETTINGS:
        raise ValueError("Unknown or modified language fixture specification")
    if set(manifest["files"]) != {"dev.jsonl", "test.jsonl", "rubric.json"}:
        raise ValueError("Incomplete language fixture manifest")
    for name, sha in manifest["files"].items():
        if digest(destination / name) != sha:
            raise ValueError("Frozen language fixture changed")
    rows = [
        json.loads(line)
        for split in ("dev", "test")
        for line in (destination / f"{split}.jsonl").read_text().splitlines()
    ]
    expected = fixture_rows()
    # Detect semantic/schema alterations even if an operator rehashes the manifest.
    if rows != expected or json.loads((destination / "rubric.json").read_text()) != RUBRIC:
        raise ValueError("Language fixtures differ from the versioned implementation")
    return rows


def protected_texts(destination: Path) -> list[str]:
    """All authored texts, including references, must be excluded from new training."""
    return sorted(
        {
            text
            for row in load_fixtures(destination)
            for text in [
                row["prompt"],
                row["passage"],
                row["question"],
                row["reference"],
                *row["required_facts"],
            ]
            if text
        }
    )


def evaluate(
    root: Path,
    fixtures: Path,
    checkpoint: Path,
    output: Path,
    seconds: int = 5400,
    device: str = "cuda",
    lease_root: Path | None = None,
) -> dict[str, Any]:
    """Generate/resume without training, sharing a persistent four-hour pair budget.

    ``root`` owns the storage policy and GPU lock; all language evaluation outputs
    must lie inside its benchmarks directory. Existing evaluation budgets are never
    altered. The combined language budget is persistent across both checkpoints.
    """
    import fcntl

    from codexa.data.v2_storage import Storage

    if not 0 < seconds <= 14400 or device not in {"cpu", "cuda"}:
        raise ValueError("Bounded positive evaluation time and explicit CPU/CUDA device required")
    if not output.resolve().is_relative_to((root / "benchmarks").resolve()):
        raise ValueError("Language evaluation outputs must remain under root/benchmarks")
    Storage.load(root).check(64 * 1024**2)
    (root / "tmp").mkdir(exist_ok=True)
    with ((lease_root or root) / "tmp/gpu-evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation owns the GPU lease") from None
        return _evaluate(root, fixtures, checkpoint, output, seconds, device)


def _evaluate(
    root: Path, fixtures: Path, checkpoint: Path, output: Path, seconds: int, device: str
) -> dict[str, Any]:
    import torch

    from codexa.evaluation.benchmark import load_frozen, model_hash, repetition
    from codexa.inference.generate import generate
    from codexa.safety import ATTEMPTS

    items = load_fixtures(fixtures)
    checkpoint_sha = digest(checkpoint)
    identity = {
        "checkpoint_sha256": checkpoint_sha,
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "settings": SETTINGS,
        "device": device,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Evaluation resume identity changed")
    else:
        if any(output.iterdir()):
            raise ValueError("Output has artifacts without a valid evaluation identity")
        atomic(identity_path, identity)
    budget_path = root / "benchmarks/language-pair-budget.json"
    budget = (
        json.loads(budget_path.read_text())
        if budget_path.exists()
        else {"limit_seconds": 14400, "used_seconds": 0.0}
    )
    if budget.get("limit_seconds") != 14400:
        raise ValueError("Combined language evaluation budget changed")
    if budget.get("active_since"):
        budget["used_seconds"] += max(0, time.time() - budget.pop("active_since"))
    limit = min(seconds, 14400 - budget["used_seconds"])
    if limit <= 0:
        atomic(budget_path, budget)
        raise RuntimeError("Combined language evaluation budget exhausted")
    records_path = output / "outputs.jsonl"
    prior = (
        [json.loads(line) for line in records_path.read_text().splitlines()]
        if records_path.exists()
        else []
    )
    allowed = {r["id"] + ":" + setting for r in items for setting in ("greedy", "sample")}
    seen = {r["key"] for r in prior}
    if len(seen) != len(prior) or not seen <= allowed:
        raise ValueError("Duplicate or foreign resumed generation records")
    if prior:
        _read_outputs(output)
    tokenizer_sha: str | None = None
    before: str | None = None
    model = None
    started = time.monotonic()
    atomic(budget_path, budget | {"active_since": time.time()})
    complete = False
    result: dict[str, Any] = {}
    try:
        model, tok, _state = load_frozen(checkpoint, device)
        tokenizer_sha = hashlib.sha256(_state["tokenizer_json"].encode()).hexdigest()
        before = model_hash(model)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with records_path.open("a") as handle:
            for item in items:
                for setting in ("greedy", "sample"):
                    key = item["id"] + ":" + setting
                    if key in seen:
                        continue
                    if time.monotonic() - started >= limit:
                        break
                    prefix = [2, *tok.encode(item["prompt"])]
                    ids = generate(model, prefix, greedy=setting == "greedy", **SETTINGS)[
                        len(prefix) :
                    ]
                    row = {
                        "key": key,
                        "item_id": item["id"],
                        "split": item["split"],
                        "task": item["task"],
                        "family": item["family"],
                        "setting": setting,
                        "prompt": item["prompt"],
                        "ids": ids,
                        "text": tok.decode(ids),
                        "settings": SETTINGS,
                        "stop_reason": "eos" if ids and ids[-1] == 3 else "token_limit",
                        "truncated": not bool(ids and ids[-1] == 3),
                        "rubric_status": "unreviewed",
                        **repetition(ids),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    seen.add(key)
                    if len(seen) % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "records": len(seen),
                                    "expected": len(allowed),
                                    "elapsed_seconds": time.monotonic() - started,
                                }
                            ),
                            flush=True,
                        )
                if time.monotonic() - started >= limit:
                    break
            complete = seen == allowed
    finally:
        elapsed = time.monotonic() - started
        budget["used_seconds"] += elapsed
        atomic(budget_path, budget)
        unchanged = model is not None and before is not None and before == model_hash(model)
        result = {
            "complete": complete,
            "records": len(seen),
            "expected_records": len(allowed),
            **identity,
            "elapsed_seconds": elapsed,
            "weights_unchanged": unchanged,
            "tokenizer_sha256": tokenizer_sha,
            "outputs_sha256": digest(records_path) if records_path.exists() else None,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "forbidden_attempts": list(ATTEMPTS),
            "budget": budget,
            "human_readiness": "unreviewed; not established",
        }
        if device == "cuda" and model is not None:
            result.update(
                gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved=torch.cuda.max_memory_reserved(),
            )
        atomic(output / "verification.json", result)
        if model is not None and (not unchanged or ATTEMPTS):
            raise RuntimeError("Forward-only model preservation failed")
    return result


def _read_outputs(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require saved generation integrity and forward-only verification."""
    identity = json.loads((output / "identity.json").read_text())
    verification = json.loads((output / "verification.json").read_text())
    if (
        identity.get("settings") != SETTINGS
        or verification.get("outputs_sha256") != digest(output / "outputs.jsonl")
        or any(verification.get(key) != value for key, value in identity.items())
        or verification.get("weights_unchanged") is not True
        or verification.get("backward_calls") != 0
        or verification.get("optimizer_updates") != 0
        or verification.get("forbidden_attempts") != []
    ):
        raise ValueError("Generation evidence changed or lacks forward-only verification")
    rows = [json.loads(line) for line in (output / "outputs.jsonl").read_text().splitlines()]
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate generation identity")
    if verification.get("records") != len(rows):
        raise ValueError("Generation record count changed")
    return identity, rows


def review_packet(
    fixtures: Path,
    outputs: list[Path],
    destination: Path,
    seed: int = 42,
    split: str | None = None,
    setting: str | None = None,
) -> dict[str, Any]:
    """Blind checkpoint identity, retaining references for independent rubric review."""
    if split not in {None, "dev", "test"} or setting not in {None, "greedy", "sample"}:
        raise ValueError("Unknown review split or decoding setting")
    items = {row["id"]: row for row in load_fixtures(fixtures)}
    if len({p.resolve() for p in outputs}) != len(outputs) or not outputs:
        raise ValueError("Distinct nonempty checkpoint output directories required")
    pairs = []
    for output in outputs:
        identity, records = _read_outputs(output)
        if identity["fixture_manifest_sha256"] != digest(fixtures / "manifest.json"):
            raise ValueError("Outputs use different fixtures")
        for record in records:
            item = items[record["item_id"]]
            if record["key"] != item["id"] + ":" + record["setting"] or record["setting"] not in {
                "greedy",
                "sample",
            }:
                raise ValueError("Generation case or decoding identity changed")
            if (split is None or item["split"] == split) and (
                setting is None or record["setting"] == setting
            ):
                pairs.append((output, identity, record, item))
    if not pairs:
        raise ValueError("No generation records match the preregistered review selection")
    random.Random(seed).shuffle(pairs)
    destination.mkdir(parents=True, exist_ok=False)
    blind, mapping = [], []
    for i, (output, identity, record, item) in enumerate(pairs):
        ident = f"L{i:05d}"
        blind.append(
            {
                "id": ident,
                "prompt": item["prompt"],
                "reference": item["reference"],
                "required_facts": item["required_facts"],
                "text": record["text"],
                "truncated": record["truncated"],
                "rubric": RUBRIC,
            }
        )
        mapping.append(
            {
                "id": ident,
                "output": str(output.resolve()),
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "key": record["key"],
                "task": item["task"],
                "split": item["split"],
            }
        )
    for name, rows in (("blind.jsonl", blind), ("mapping.jsonl", mapping)):
        (destination / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    manifest = {
        "version": VERSION,
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "records": len(blind),
        "selection": {
            "split": split,
            "setting": setting,
            "rule": "All saved cases matching fixed split/setting filters; unselected generations remain unreviewed.",
        },
        "seed": seed,
        "files": {name: digest(destination / name) for name in ("blind.jsonl", "mapping.jsonl")},
        "inputs": [
            {"path": str(p.resolve()), "outputs_sha256": digest(p / "outputs.jsonl")}
            for p in outputs
        ],
    }
    atomic(destination / "manifest.json", manifest)
    return manifest


def score_reviews(packet: Path, reviewer_files: dict[str, Path]) -> dict[str, Any]:
    """Require two complete independent ratings and conservatively combine them."""
    if (
        len(reviewer_files) != 2
        or any(not name.strip() for name in reviewer_files)
        or len({p.resolve() for p in reviewer_files.values()}) != 2
    ):
        raise ValueError("Two distinct independent reviewer identities and files required")
    manifest = json.loads((packet / "manifest.json").read_text())
    if manifest.get("version") != VERSION or set(manifest["files"]) != {
        "blind.jsonl",
        "mapping.jsonl",
    }:
        raise ValueError("Unknown or incomplete blinded packet specification")
    for source in manifest["inputs"]:
        if digest(Path(source["path"]) / "outputs.jsonl") != source["outputs_sha256"]:
            raise ValueError("Generation evidence changed after blinding")
    for name, sha in manifest["files"].items():
        if digest(packet / name) != sha:
            raise ValueError("Frozen blinded packet changed")
    mapping = [json.loads(line) for line in (packet / "mapping.jsonl").read_text().splitlines()]
    expected = {r["id"] for r in mapping}
    scores = []
    for path in reviewer_files.values():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError("Incomplete or duplicated independent ratings")
        for row in rows:
            if (
                set(row.get("scores", {})) != set(DIMENSIONS)
                or any(
                    type(value) is not int or value not in (0, 1, 2)
                    for value in row["scores"].values()
                )
                or type(row.get("uncertain")) is not bool
                or not row.get("reason", "").strip()
            ):
                raise ValueError(
                    "Explicit 0–2 dimension scores, uncertainty and rationale required"
                )
        scores.append({r["id"]: r for r in rows})
    combined = []
    for item in mapping:
        a, b = (s[item["id"]] for s in scores)
        combined.append(
            item
            | {
                "scores": {dim: min(a["scores"][dim], b["scores"][dim]) for dim in DIMENSIONS},
                "uncertain": a["uncertain"] or b["uncertain"],
                "disagreement": a["scores"] != b["scores"] or a["uncertain"] != b["uncertain"],
                "reasons": [a["reason"], b["reason"]],
            }
        )
    return {
        "version": VERSION,
        "packet_manifest_sha256": digest(packet / "manifest.json"),
        "reviewers": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in reviewer_files.items()
        },
        "records": combined,
        "readiness": "Scored evidence only; no automatic competence or SFT authorization",
    }


def compare(before: Path, after: Path, reviewed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pair actual records; report missing cases rather than filling them with zeros."""
    evidence = [_read_outputs(p) for p in (before, after)]
    identities = [identity for identity, _ in evidence]
    if (
        identities[0]["fixture_manifest_sha256"] != identities[1]["fixture_manifest_sha256"]
        or identities[0]["settings"] != identities[1]["settings"]
    ):
        raise ValueError("Paired evaluation requires identical fixtures and settings")
    loaded = [rows for _, rows in evidence]
    indexed = [{r["key"]: r for r in rows} for rows in loaded]
    if any(len(d) != len(rows) for d, rows in zip(indexed, loaded)):
        raise ValueError("Duplicate paired generation identity")
    paired = sorted(indexed[0].keys() & indexed[1].keys())
    groups: dict[str, list[str]] = defaultdict(list)
    for key in paired:
        row = indexed[0][key]
        groups[f"{row['split']}:{row['task']}:{row['setting']}"].append(key)
    metrics = {}
    for group, keys in groups.items():
        metrics[group] = {
            "pairs": len(keys),
            "severe_loops_before": sum(indexed[0][k]["severe_loop"] for k in keys),
            "severe_loops_after": sum(indexed[1][k]["severe_loop"] for k in keys),
            "mean_repeated_4gram_change": sum(
                indexed[1][k]["repeated_token_4gram_fraction"]
                - indexed[0][k]["repeated_token_4gram_fraction"]
                for k in keys
            )
            / len(keys),
        }
    result = {
        "pairs": len(paired),
        "before_only": sorted(indexed[0].keys() - indexed[1].keys()),
        "after_only": sorted(indexed[1].keys() - indexed[0].keys()),
        "metrics": metrics,
        "rubric_status": "unreviewed",
        "limitation": "Paired observed cases; correlated scenario tasks and incomplete cases limit conclusions.",
    }
    if reviewed is not None:
        by = {(r["checkpoint_sha256"], r["key"]): r for r in reviewed["records"]}
        deltas: dict[str, list[int]] = defaultdict(list)
        for key in paired:
            a, b = (by.get((identity["checkpoint_sha256"], key)) for identity in identities)
            if a is not None and b is not None and not (a["uncertain"] or b["uncertain"]):
                for dimension in DIMENSIONS:
                    deltas[dimension].append(b["scores"][dimension] - a["scores"][dimension])
        result["rubric_status"] = "independently reviewed matched records; uncertain cases excluded"
        result["rubric_deltas"] = {
            dimension: {"n": len(values), "mean_change": sum(values) / len(values)}
            for dimension, values in deltas.items()
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "packet", "score", "compare"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lease-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--reviewer", action="append", default=[], help="Independent identity:path")
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--setting", choices=("greedy", "sample"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.fixtures)
    elif args.action == "evaluate":
        if args.root is None or args.checkpoint is None:
            parser.error("evaluate requires --root and --checkpoint")
        result = evaluate(
            args.root,
            args.fixtures,
            args.checkpoint,
            args.output,
            args.seconds,
            args.device,
            args.lease_root,
        )
    elif args.action == "packet":
        result = review_packet(
            args.fixtures, args.inputs, args.output, split=args.split, setting=args.setting
        )
    elif args.action == "score":
        reviewers = dict(value.split(":", 1) for value in args.reviewer)
        result = score_reviews(
            args.fixtures, {name: Path(path) for name, path in reviewers.items()}
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic(args.output, result)
    else:
        if len(args.inputs) != 2:
            parser.error("compare requires exactly two --inputs")
        result = compare(*args.inputs)
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
