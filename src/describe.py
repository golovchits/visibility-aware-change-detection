"""One plain sentence per object, built only from the numbers in results.json.

"same as video 1" and "moved to" objects get no sentence of their own. Their partner's sentence covers them.
"""


def evidence(r, other):
    """What the other video saw of this object's spot."""
    empty = 1 - r["present_fraction"]
    return (f"Video {other} saw {r['coverage']:.0%} of that spot from {len(r['checked_by'])} views, "
            f"and {empty:.0%} of what it saw was empty.")


def sentence(r, records, pair, config):
    first, second = pair
    other = second if r["video"] == 1 else first
    label = r["label"]
    if r["state"] == "unchanged":
        return (f"The {label} is unchanged. Video {second} saw {r['coverage']:.0%} of it from {len(r['checked_by'])} "
                f"views, and {r['present_fraction']:.0%} of what it saw was the same object.")
    if r["state"] == "removed":
        return f"The {label} is gone. " + evidence(r, other)
    if r["state"] == "added":
        return f"A {label} appeared. " + evidence(r, other)
    if r["state"] == "moved":
        new = records[r["moved_to"]]
        return (f"The {label} moved {r['moved_cm']:.0f} cm. Video {second} saw {r['coverage']:.0%} of its old spot "
                f"and {1 - r['present_fraction']:.0%} of that was empty. Video {first} saw {new['coverage']:.0%} "
                f"of its new spot and {1 - new['present_fraction']:.0%} of that was empty.")
    if r["state"] == "not checked":
        where = "" if r["video"] == 1 else f" in video {second}"
        return (f"The {label}{where} was not checked. Video {other} saw only {r['coverage']:.0%} of it, "
                f"below the {config['c_min']:.0%} needed to decide.")
    return None


def describe(records, pair, config):
    """Sentences in a fixed order: changes first, then not checked, then unchanged."""
    order = ["removed", "added", "moved", "not checked", "unchanged"]
    lines = []
    for state in order:
        for r in records:
            if r["state"] == state:
                lines.append((r["id"], sentence(r, records, pair, config)))
    return lines
