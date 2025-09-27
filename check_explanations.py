import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
ANSWER_PATH = BASE / "answer_sheetappper.json"
EXPL_PATH = BASE / "ExplanationBankappper.json"


def normalize_qid(qid: str) -> str:
    # Convert Q01 -> Q1, Q1 stays Q1, Q10 stays Q10
    m = re.match(r"Q0*(\d+)$", qid)
    return f"Q{m.group(1)}" if m else qid


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: failed to load {p}: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    answer = load_json(ANSWER_PATH)
    expl = load_json(EXPL_PATH)

    required = set()
    for q in answer.get("questions", []):
        qid = normalize_qid(q.get("questionId", ""))
        for opt in q.get("options", []):
            level = (opt.get("level") or "").strip()
            label = (opt.get("label") or "").strip()
            if not label:
                continue
            # If level is "Advanced" then we do NOT require explanation per spec
            if level.lower() != "advanced":
                required.add((qid, label))

    available = set()
    for entry in expl:
        qid = normalize_qid(entry.get("questionId", ""))
        label = (entry.get("option") or "").strip()
        if label:
            available.add((qid, label))

    missing = sorted(required - available)

    if not missing:
        print("OK: All non-Advanced options have at least one explanation entry in ExplanationBank.")

        # Additional check: Show profile coverage
        print("\nPROFILE COVERAGE ANALYSIS:")
        profiles_covered = {}
        for entry in expl:
            qid = normalize_qid(entry.get("questionId", ""))
            label = entry.get("option", "")
            profile = entry.get("profile", {})
            key = (qid, label)
            if key not in profiles_covered:
                profiles_covered[key] = []
            profile_str = f"{profile.get('gender', '')}-{profile.get('proficiency', '')}-{profile.get('education', '')}"
            profiles_covered[key].append(profile_str)

        print(
            f"Total (Question, Option) pairs with explanations: {len(profiles_covered)}")
        for key, profiles in profiles_covered.items():
            qid, label = key
            print(f"  {qid} option {label}: {len(profiles)} profile variations")

        return 0

    print("MISSING EXPLANATIONS:")
    for qid, label in missing:
        print(f"- {qid} option {label}")

    print()
    print("Summary:")
    print(f"  required non-Advanced pairs : {len(required)}")
    print(f"  available explanation pairs : {len(available)}")
    print(f"  missing                     : {len(missing)}")
    print()
    print("Please add explanation entries for the missing (questionId, option) pairs into ExplanationBankappper.json.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
