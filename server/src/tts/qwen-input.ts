/* Bound Qwen's autoregressive input, not its output: oversized passages can
   run to the decoder's token ceiling. Keep every character at this provider
   boundary so the manuscript and the caller's sentence identities stay intact. */
const QWEN_INPUT_CHARACTERS = 600;

export function splitQwenInput(text: string): string[] {
  const characters = Array.from(text);
  if (characters.length <= QWEN_INPUT_CHARACTERS) return [text];

  const pieces: string[] = [];
  let offset = 0;
  while (characters.length - offset > QWEN_INPUT_CHARACTERS) {
    const window = characters.slice(offset, offset + QWEN_INPUT_CHARACTERS).join('');
    let end = 0;
    /* Prefer natural breaks in the latter half to avoid tiny requests. Keep
       whitespace with the preceding piece; never trim or reconstruct text. */
    for (const boundary of [
      /\r?\n[\t ]*\r?\n/gu,
      /[。！？]["'”’»\])]*\s*|[.!?]["'”’»\])]*(?:\s+|$)/gu,
      /\s+/gu,
    ]) {
      for (const match of window.matchAll(boundary)) {
        const candidate = Array.from(window.slice(0, match.index + match[0].length)).length;
        if (candidate >= QWEN_INPUT_CHARACTERS / 2) end = candidate;
      }
      if (end) break;
    }
    /* An early whitespace break is still better than cutting through a word.
       Unspaced text falls back to a Unicode code-point boundary. */
    if (!end) {
      for (const match of window.matchAll(/\s+/gu)) {
        end = Array.from(window.slice(0, match.index + match[0].length)).length;
      }
    }
    end ||= QWEN_INPUT_CHARACTERS;
    pieces.push(characters.slice(offset, offset + end).join(''));
    offset += end;
  }
  pieces.push(characters.slice(offset).join(''));
  /* The endpoint rejects blank text after trimming. Attach blank-only pieces
     to a spoken neighbor, preserving even pathological whitespace runs. Only
     that whitespace can exceed the bound; ordinary normalized inputs cannot. */
  const spoken: string[] = [];
  let leadingWhitespace = '';
  for (const piece of pieces) {
    if (piece.trim()) {
      spoken.push(leadingWhitespace + piece);
      leadingWhitespace = '';
    } else if (spoken.length) {
      spoken[spoken.length - 1] += piece;
    } else {
      leadingWhitespace += piece;
    }
  }
  return spoken.length ? spoken : [text];
}
