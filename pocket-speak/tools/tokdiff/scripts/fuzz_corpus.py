"""Write N random mixed-script strings as JSON lines. Usage: fuzz_corpus.py OUT N SEED"""

import json
import random
import sys

out, count, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
random.seed(seed)
pools = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    " \t\n\u00a0\u2003\u200b\u3000",
    ".,;:!?'\"()[]{}<>/\\|@#$%^&*_-+=~`…—–“”‘’«»¿¡",
    "éèêëàâäôöûüçñßøåæœÉÀÇÑ",
    "αβγδεζηθλμπσωΑΒΓΔ",
    "абвгдежзийклмнопрстуфхцчшщыэюя",
    "日本語中文한국어ひらがなカタカナ",
    "😀👍🏽🎉🚀❤️🇺🇸👨‍👩‍👧",
    "ＡＢＣａｂｃ１２３①②③ﬁﬂ㎞℃™",
    "\u0301\u0308\u200d\ufe0f",
]
with open(out, "w") as f:
    for _ in range(count):
        length = random.randint(1, 80)
        text = "".join(random.choice(random.choice(pools)) for _ in range(length))
        f.write(json.dumps(text, ensure_ascii=True) + "\n")
