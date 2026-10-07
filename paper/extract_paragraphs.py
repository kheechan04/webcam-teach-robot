"""main.tex를 문단 단위 평문으로 뽑는다 (숫자 매크로를 채움). 한국어 대조판(make_ko.py)이 쓴다.

    uv run python paper/extract_paragraphs.py   → 문단 번호와 내용을 출력
"""

import re
from pathlib import Path

P = Path(__file__).resolve().parent

SIMPLE = {r"\condI{}": "C1", r"\condII{}": "C2", r"\condIII{}": "C3", r"\condIV{}": "C4", r"\condVz{}": "C4₀",
          r"\condV{}": "C5", r"\condI": "C1", r"\condII": "C2", r"\condIII": "C3", r"\condIV": "C4", r"\condVz": "C4₀",
          r"\condV": "C5", r"\textminus": "−", "~": " ", r"\%": "%", "$-$": "−", r"\ ": " ", "--": "–", "``": "“", "''": "”",
          r"$\to$": "→", r"\_": "_"}

MATH = [(r"\\mathbf\{(\w)\}", r"\1"), (r"\\text\{([^}]*)\}", r"\1"), (r"\^\\top", "ᵀ"), (r"\^\\circ", "°"),
        (r"\\times", "×"), (r"10\^\{-5\}", "10⁻⁵"), (r"\\exp", "exp"), (r"\\log", "log")]


def macros() -> dict[str, str]:
    out = {}
    for line in (P / "numbers.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2).replace(r"\textminus", "−")
    return out


def plain(s: str, mac: dict[str, str]) -> str:
    s = re.sub(r"\\todo\{([^}]*)\}", r"[TODO: \1]", s)
    for a, b in MATH:
        s = re.sub(a, b, s)
    for k in sorted(mac, key=len, reverse=True):
        s = re.sub(r"\\" + k + r"(?![A-Za-z])(\{\})?", lambda _m, v=mac[k]: v, s)
    for k in sorted(SIMPLE, key=len, reverse=True):
        s = s.replace(k, SIMPLE[k])
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", "[ref]", s)
    s = re.sub(r"(Section|Table|Fig\.|Figure|Appendix)\s*\\ref\{[^}]*\}", r"\1", s)
    s = re.sub(r"\\(emph|textbf|paragraph|texttt|url)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = s.replace(r"\item", "•").replace("$", "")
    return re.sub(r"\s+", " ", s).strip()


def paragraphs() -> list[tuple[str, str]]:
    """(종류, 내용) 목록. 종류: section, para. 표·그림은 빼고, 부록 본문은 넣는다."""
    tex = (P / "main.tex").read_text(encoding="utf-8")
    mac = macros()
    body = (tex[tex.index(r"\begin{document}"):tex.index(r"\bibliographystyle")] + "\n\n"
            + tex[tex.index(r"\appendix") + len(r"\appendix"):tex.index(r"\end{document}")])
    body = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", "", body, flags=re.S)
    body = re.sub(r"(?m)^%.*$", "", body)
    out = []
    for block in re.split(r"\n\s*\n", body):
        b = block.replace(r"\begin{document}", "").replace(r"\maketitle", "").strip()
        for env in ("abstract", "itemize", "enumerate"):
            b = b.replace(rf"\begin{{{env}}}", "").replace(rf"\end{{{env}}}", "")
        b = b.strip()
        m = re.match(r"\\section\*?\{([^}]*)\}\s*(.*)", b, flags=re.S)
        if m:
            out.append(("section", plain(m.group(1), mac)))
            b = m.group(2).strip()
        m = re.match(r"\\paragraph\{([^}]*)\}\s*(.*)", b, flags=re.S)
        if m:
            out.append(("para", "**" + plain(m.group(1), mac) + "** " + plain(m.group(2), mac)))
        elif plain(b, mac):
            out.append(("para", plain(b, mac)))
    return out


if __name__ == "__main__":
    for i, (k, t) in enumerate(paragraphs()):
        print(f"[{i}] ({k}) {t}\n")
