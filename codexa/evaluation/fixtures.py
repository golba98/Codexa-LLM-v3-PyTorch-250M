"""Original CC0 language/math evaluation fixtures, independent of acquired text."""

import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest


def identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def facts() -> list[tuple[str, str, str, str]]:
    result: list[tuple[str, str, str, str]] = []
    countries = "France:Paris;Japan:Tokyo;Italy:Rome;Germany:Berlin;Spain:Madrid;Portugal:Lisbon;Norway:Oslo;Sweden:Stockholm;Finland:Helsinki;Denmark:Copenhagen;Iceland:Reykjavik;Ireland:Dublin;United Kingdom:London;Poland:Warsaw;Austria:Vienna;Hungary:Budapest;Greece:Athens;Canada:Ottawa;Brazil:Brasilia;Argentina:Buenos Aires;Chile:Santiago;Peru:Lima;Colombia:Bogota;Mexico:Mexico City;Egypt:Cairo;Kenya:Nairobi;Ghana:Accra;Nigeria:Abuja;Australia:Canberra;New Zealand:Wellington"
    for pair in countries.split(";"):
        country, city = pair.split(":")
        result.append(
            (
                f"The national capital of {country} is",
                city,
                f"Which city is the national capital of {country}?",
                "https://en.wikipedia.org/wiki/" + country.replace(" ", "_"),
            )
        )
    opposites = "hot:cold;day:night;above:below;inside:outside;light:dark;early:late;empty:full;rough:smooth;wet:dry;open:closed;up:down;left:right;north:south;east:west;fast:slow;heavy:light;wide:narrow;young:old;thick:thin;hard:soft"
    for pair in opposites.split(";"):
        word, answer = pair.split(":")
        result.append(
            (
                f"A common opposite of {word} is",
                answer,
                f"Which common word means the opposite of {word}?",
                "original lexical fixture",
            )
        )
    elements = "hydrogen:H;helium:He;lithium:Li;beryllium:Be;boron:B;carbon:C;nitrogen:N;oxygen:O;fluorine:F;neon:Ne;sodium:Na;magnesium:Mg;aluminium:Al;silicon:Si;phosphorus:P;sulfur:S;chlorine:Cl;argon:Ar;potassium:K;calcium:Ca;iron:Fe;copper:Cu;silver:Ag;gold:Au;mercury:Hg"
    for pair in elements.split(";"):
        element, symbol = pair.split(":")
        result.append(
            (
                f"The chemical symbol for {element} is",
                symbol,
                f"Which chemical symbol represents {element}?",
                "https://iupac.org/what-we-do/periodic-table-of-elements/",
            )
        )
    babies = "dog:puppy;cat:kitten;cow:calf;horse:foal;sheep:lamb;goat:kid;pig:piglet;duck:duckling;goose:gosling;chicken:chick;deer:fawn;lion:cub;tiger:cub;bear:cub;elephant:calf;whale:calf;seal:pup;frog:tadpole;swan:cygnet;kangaroo:joey"
    for pair in babies.split(";"):
        animal, answer = pair.split(":")
        result.append(
            (
                f"A young {animal} is commonly called a",
                answer,
                f"What is the common name for a young {animal}?",
                "https://en.wikipedia.org/wiki/" + animal,
            )
        )
    sounds = "dog:bark;cat:meow;cow:moo;horse:neigh;sheep:bleat;goat:bleat;pig:grunt;duck:quack;goose:honk;lion:roar;owl:hoot;bee:buzz;snake:hiss;frog:croak;wolf:howl;donkey:bray;elephant:trumpet;mouse:squeak;pigeon:coo;crow:caw"
    for pair in sounds.split(";"):
        animal, answer = pair.split(":")
        result.append(
            (
                f"A typical sound made by {'an' if animal[0] in 'aeiou' else 'a'} {animal} is a",
                answer,
                f"What typical sound does a {animal} make?",
                "original common-vocabulary fixture",
            )
        )
    units = [
        ("minute", "seconds", 60),
        ("hour", "minutes", 60),
        ("day", "hours", 24),
        ("week", "days", 7),
        ("year", "months", 12),
        ("decade", "years", 10),
        ("century", "years", 100),
        ("millennium", "years", 1000),
        ("meter", "centimeters", 100),
        ("kilometer", "meters", 1000),
        ("centimeter", "millimeters", 10),
        ("kilogram", "grams", 1000),
        ("gram", "milligrams", 1000),
        ("liter", "milliliters", 1000),
        ("hectare", "square meters", 10000),
        ("metric tonne", "kilograms", 1000),
        ("dozen", "items", 12),
        ("pair", "items", 2),
        ("score", "items", 20),
        ("gross", "items", 144),
    ]
    for unit, smaller, value in units:
        result.append(
            (
                f"One {unit} contains this many {smaller}:",
                str(value),
                f"How many {smaller} are in one {unit}?",
                "https://www.bipm.org/en/measurement-units",
            )
        )
    shapes = [
        ("triangle", "sides", 3),
        ("quadrilateral", "sides", 4),
        ("pentagon", "sides", 5),
        ("hexagon", "sides", 6),
        ("heptagon", "sides", 7),
        ("octagon", "sides", 8),
        ("nonagon", "sides", 9),
        ("decagon", "sides", 10),
        ("square", "right angles", 4),
        ("rectangle", "right angles", 4),
        ("circle", "corners", 0),
        ("triangle", "angles", 3),
        ("cube", "faces", 6),
        ("cube", "edges", 12),
        ("cube", "vertices", 8),
        ("tetrahedron", "faces", 4),
        ("tetrahedron", "edges", 6),
        ("tetrahedron", "vertices", 4),
        ("rectangular prism", "faces", 6),
        ("square pyramid", "faces", 5),
    ]
    for shape, attribute, value in shapes:
        result.append(
            (
                f"The number of {attribute} on a {shape} is",
                str(value),
                f"How many {attribute} does a {shape} have?",
                "original elementary-geometry fixture",
            )
        )
    import calendar

    for month in range(1, 13):
        name = calendar.month_name[month]
        value = calendar.monthrange(2023, month)[1]
        result.append(
            (
                f"In a non-leap year, the number of days in {name} is",
                str(value),
                f"How many days does {name} have in a non-leap year?",
                "Python calendar, Gregorian year 2023",
            )
        )
    physical = [
        ("The planet nearest the Sun is", "Mercury"),
        ("The largest planet is", "Jupiter"),
        ("The planet between Earth and Jupiter is", "Mars"),
        ("Earth's natural satellite is the", "Moon"),
        ("The star at the center of our solar system is the", "Sun"),
        ("The force pulling objects toward Earth is called", "gravity"),
        ("Water in its solid form is called", "ice"),
        ("Liquid water changing into gas is called", "evaporation"),
        ("Water vapor changing into liquid is called", "condensation"),
        ("A solid changing into a liquid is called", "melting"),
        ("The gas plants take in during photosynthesis is", "carbon dioxide"),
        ("The gas released by plants during photosynthesis is", "oxygen"),
        ("The process plants use to make sugars using light is", "photosynthesis"),
        ("The organ that pumps blood is the", "heart"),
        ("The organs primarily used to exchange gases while breathing are the", "lungs"),
        ("The organ that controls much of the nervous system is the", "brain"),
        ("The bones inside the body together form the", "skeleton"),
        ("The ocean between Africa and South America is the", "Atlantic Ocean"),
        ("The largest ocean on Earth is the", "Pacific Ocean"),
        ("The continent containing Egypt is", "Africa"),
        ("The continent containing Brazil is", "South America"),
        ("The continent containing Japan is", "Asia"),
        ("The continent containing France is", "Europe"),
        ("An animal that eats only plants is called a", "herbivore"),
        ("An animal that eats both plants and animals is called an", "omnivore"),
        ("An animal that eats other animals is called a", "carnivore"),
        ("A person who studies rocks is called a", "geologist"),
        ("A person who studies the stars is called an", "astronomer"),
        ("A person who studies plants is called a", "botanist"),
        ("A person who studies the weather is called a", "meteorologist"),
        ("An instrument used to measure atmospheric pressure is a", "barometer"),
        ("The standard unit of electrical resistance is the", "ohm"),
        ("The standard unit of electrical current is the", "ampere"),
    ]
    result.extend(
        (
            p,
            a,
            "Complete this factual association: " + p + "?",
            "original elementary-science fixture; verify before readiness claims",
        )
        for p, a in physical
    )
    assert len(result) >= 200
    return result


GRAMMAR = [
    ("{n} walks to school.", "{n} walk to school."),
    ("{n} has a blue bag.", "{n} have a blue bag."),
    ("{n} is reading quietly.", "{n} are reading quietly."),
    ("{n} does not like rain.", "{n} do not like rain."),
    ("{n} can swim well.", "{n} can swims well."),
    ("{n} must finish the task.", "{n} must finishes the task."),
    ("{n} will arrive soon.", "{n} will arrives soon."),
    ("{n} should wait outside.", "{n} should waits outside."),
    ("{n} did not see the sign.", "{n} did not saw the sign."),
    ("{n} has written a letter.", "{n} has wrote a letter."),
    ("{n} has eaten lunch.", "{n} has ate lunch."),
    ("{n} has taken the bus.", "{n} has took the bus."),
    ("{n} is waiting for me.", "{n} is waiting for I."),
    ("I gave the book to {n} and her.", "I gave the book to {n} and she."),
    ("{n} and I went home.", "{n} and me went home."),
    ("{n} saw two birds.", "{n} saw two bird."),
    ("{n} bought three pencils.", "{n} bought three pencil."),
    ("{n} read several books.", "{n} read several book."),
    ("{n} ate an apple.", "{n} ate a apple."),
    ("{n} saw a dog.", "{n} saw an dog."),
    ("{n} opened an umbrella.", "{n} opened a umbrella."),
    ("{n} found an old coin.", "{n} found a old coin."),
    ("{n} wore a uniform.", "{n} wore an uniform."),
    ("{n} waited for an hour.", "{n} waited for a hour."),
    ("{n} enjoys reading.", "{n} enjoys to reading."),
    ("{n} wants to leave.", "{n} wants leaving to."),
    ("{n} is interested in music.", "{n} is interested on music."),
    ("{n} is good at drawing.", "{n} is good in to drawing."),
    ("{n} spoke very clearly.", "{n} spoke very clearfully."),
    ("{n} carefully opened the box.", "{n} careful opened the box."),
    ("{n} is taller than her brother.", "{n} is more taller than her brother."),
    ("{n} chose the best answer.", "{n} chose the most best answer."),
    ("{n} has many friends.", "{n} has much friends."),
    ("{n} drank some water.", "{n} drank a water much."),
    ("{n} knows where the library is.", "{n} knows where is the library."),
    ("{n} asked what time it was.", "{n} asked what time was it."),
    ("{n} is one of the students.", "{n} is one of the student."),
    ("{n} helped the children.", "{n} helped the childs."),
    ("{n} brushed her teeth.", "{n} brushed her tooths."),
    ("{n} washed her feet.", "{n} washed her foots."),
]


def freeze(root: Path) -> dict[str, Any]:
    path = root / "benchmarks" / "v3"
    path.mkdir(parents=True, exist_ok=True)
    existing = path / "manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        for name, sha in manifest["files"].items():
            if digest(path / name) != sha:
                raise ValueError("Frozen evaluation changed")
        return manifest
    scenario_path = Path(__file__).resolve().parents[2] / "tests/fixtures/language_scenarios.txt"
    vocab_path = scenario_path.with_name("vocabulary.tsv")
    scenes = list(dict.fromkeys(scenario_path.read_text().splitlines()))
    scenes.sort(key=lambda s: identity("42:" + s))
    names = ["Nora", "Maya", "Lena", "Anna", "Rosa", "Emma", "Sara", "Zara", "Tara", "Lucy"]
    vocabulary = [tuple(line.split("\t")) for line in vocab_path.read_text().splitlines()]
    vocabulary.sort(key=lambda s: identity("42:" + s[0]))
    fact_pool = facts()
    fact_pool.sort(key=lambda s: identity("42:" + s[0]))
    # Keep offspring/sound questions for each animal and attributes of each solid together.
    import re

    groups: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in fact_pool:
        prompt = row[0]
        animal = re.search(r"(?:young |made by (?:a|an) )(\w+)", prompt)
        solid = re.search(r"on a (.+) is$", prompt)
        family = "animal:" + animal[1] if animal else "shape:" + solid[1] if solid else prompt
        groups.setdefault(family, []).append(row)
    dev_facts: list[tuple[str, str, str, str]] = []
    test_facts: list[tuple[str, str, str, str]] = []
    for family in sorted(groups, key=lambda x: identity("42:" + x)):
        group = groups[family]
        (dev_facts if len(dev_facts) + len(group) <= 100 else test_facts).extend(group)
    assert len(dev_facts) == len(test_facts) == 100
    fact_pool = dev_facts + test_facts
    items: list[dict[str, Any]] = []
    for split, offset in (("dev", 0), ("test", 100)):
        for i, scene in enumerate(scenes[offset : offset + 100]):
            for kind, prompt in (("sentence", scene), ("paragraph", scene)):
                items.append(
                    {
                        "id": identity(kind + scene),
                        "split": split,
                        "kind": kind,
                        "family": identity(scene),
                        "prompt": prompt,
                        "source": "Original CC0 scenario",
                        "scoring": "rubric_review_required",
                    }
                )
            # A closed-world relationship, without a conversational role template.
            entities = (names[i % 10], names[(i + 3) % 10])
            place = ["garden", "library", "station", "kitchen", "workshop"][i % 5]
            objects = [
                "notebook",
                "pencil",
                "scarf",
                "hat",
                "glove",
                "wallet",
                "card",
                "camera",
                "cup",
                "toy",
                "book",
                "spoon",
                "brush",
                "towel",
                "bag",
                "bottle",
                "ball",
                "ruler",
                "envelope",
                "map",
            ]
            obj = objects[i % len(objects)]
            place = ["garden", "library", "station", "kitchen", "workshop"][i // 20]
            destination = ["classroom", "office", "hall", "study", "bedroom"][i % 5]
            color = "red" if split == "dev" else "blue"
            prompt = f"{entities[0]} left a {color} {obj} in the {place}. {entities[1]} took it to a {destination}. The {obj} is now in the"
            items.append(
                {
                    "id": identity(split + str(i) + "context"),
                    "split": split,
                    "kind": "context",
                    "family": split + f":context:{i}",
                    "prompt": prompt,
                    "answer": destination,
                    "scoring": "contextual_answer_and_contradiction_review",
                    "source": "Original CC0 closed-world fixture",
                }
            )
        for i, (word, definition) in enumerate(vocabulary[offset : offset + 100]):
            foils = [x[0] for x in vocabulary if x[0] != word]
            random.Random(42 + offset + i).shuffle(foils)
            items.append(
                {
                    "id": identity("vocab:" + word),
                    "split": split,
                    "kind": "vocabulary",
                    "family": "word:" + word,
                    "prompt": f'The word for "{definition}" is',
                    "question": f"Which word names {definition}?",
                    "answer": word,
                    "candidates": [word, *foils[:3]],
                    "source": "Original CC0 definition",
                    "scoring": "candidate_ranking",
                }
            )
        for i, (prompt, answer, question, source) in enumerate(fact_pool[offset : offset + 100]):
            alternatives = sorted({x[1] for x in fact_pool if x[1] != answer})
            random.Random(42 + offset + i).shuffle(alternatives)
            items.append(
                {
                    "id": identity("fact:" + prompt),
                    "split": split,
                    "kind": "fact",
                    "family": identity(prompt),
                    "prompt": prompt,
                    "question": question,
                    "answer": answer,
                    "candidates": [answer, *alternatives[:3]],
                    "source": source,
                    "scoring": "answer_only_ranking_and_free_generation_separate",
                }
            )
        rule_offset = 0 if split == "dev" else 20
        for rule, (good, bad) in enumerate(GRAMMAR[rule_offset : rule_offset + 20], rule_offset):
            for name in names:
                items.append(
                    {
                        "id": identity(good + name),
                        "split": split,
                        "kind": "grammar",
                        "family": f"grammar-rule:{rule}",
                        "prompt": "",
                        "answer": good.format(n=name),
                        "candidates": [good.format(n=name), bad.format(n=name)],
                        "source": "Original CC0 minimal pair; standard written English",
                        "scoring": "paired_sentence_likelihood",
                    }
                )
        rng = random.Random(42 if split == "dev" else 2026)
        used: set[str] = set()
        for topic in range(6):
            for i in range(50):
                while True:
                    lo, hi = (2, 20) if split == "dev" else (21, 60)
                    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
                    if topic == 0:
                        q, answer_value, grade = f"Calculate {a} + {b}.", Fraction(a + b), 1
                    elif topic == 1:
                        q, answer_value, grade = (
                            f"A box holds {a} pencils. There are {b} boxes. How many pencils are there?",
                            Fraction(a * b),
                            3,
                        )
                    elif topic == 2:
                        q, answer_value, grade = (
                            f"Simplify the fraction {a}/{b}.",
                            Fraction(a, b),
                            5,
                        )
                    elif topic == 3:
                        x = rng.randint(lo, hi)
                        q, answer_value, grade = (
                            f"Solve for x: {a}*x + {b} = {a * x + b}.",
                            Fraction(x),
                            8,
                        )
                    elif topic == 4:
                        q, answer_value, grade = (
                            f"A rectangle has length {a} and width {b}. What is its area?",
                            Fraction(a * b),
                            5,
                        )
                    else:
                        q, answer_value, grade = (
                            f"If f(x) = {a}*x^2 + {b}, calculate f(3).",
                            Fraction(a * 9 + b),
                            10,
                        )
                    key = f"{topic}:{min(a, b)}:{max(a, b)}:" + (q if topic == 3 else "")
                    if key not in used:
                        used.add(key)
                        break
                items.append(
                    {
                        "id": identity(split + key),
                        "split": split,
                        "kind": "math",
                        "family": split + key,
                        "prompt": q + "\nThe answer is",
                        "question": q,
                        "answer": str(answer_value),
                        "topic": [
                            "arithmetic",
                            "word_problem",
                            "fraction",
                            "algebra",
                            "geometry",
                            "function",
                        ][topic],
                        "grade": grade,
                        "source": "Original CC0 deterministic generator; exact Fraction arithmetic",
                        "scoring": "numeric_answer; no full-curriculum claim",
                    }
                )
    for split in ("dev", "test"):
        selected = [x for x in items if x["split"] == split]
        (path / (split + ".jsonl")).write_text(
            "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in selected)
        )
    counts = {
        s: {
            k: sum(x["kind"] == k and x["split"] == s for x in items)
            for k in sorted({x["kind"] for x in items})
        }
        for s in ("dev", "test")
    }
    manifest = {
        "version": 3,
        "seed": 42,
        "license": "CC0-1.0 original fixtures",
        "files": {s + ".jsonl": digest(path / (s + ".jsonl")) for s in ("dev", "test")},
        "counts": counts,
        "scenario_source_sha256": digest(scenario_path),
        "vocabulary_source_sha256": digest(vocab_path),
        "grammar_rule_families_per_split": 20,
        "context_template_warning": "100 unique cases/split share one relationship template; do not count them as 100 independent paradigms",
        "readiness": "unrated; no automatic claim of human-rated language readiness",
    }
    atomic(existing, manifest)
    return manifest
