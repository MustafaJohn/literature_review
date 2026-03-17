from typing import List, Tuple


def chunk_text(text: str, max_words: int = 200) -> List[Tuple[int, str]]:
    words    = text.split()
    chunks   = []
    chunk_id = 0
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append((chunk_id, chunk))
        chunk_id += 1
    return chunks