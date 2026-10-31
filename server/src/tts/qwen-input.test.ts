import { describe, expect, it } from 'vitest';
import { splitQwenInput } from './qwen-input.js';

describe('splitQwenInput', () => {
  it.each(['', 'A short sentence.', 'Я'.repeat(600), '😀'.repeat(600)])(
    'preserves an input at or below the bound (%#)',
    (text) => {
      expect(splitQwenInput(text)).toEqual([text]);
    },
  );
  it.each([
    '  «Он сказал: „Да!“»\r\n\r\n— Нет, — ответила она.\n\n'.repeat(65),
    '日本語の文です。次の文です！'.repeat(100),
    'word '.repeat(400),
    '😀'.repeat(1201),
    'a'.repeat(1201),
  ])('keeps every code point, punctuation and whitespace within bounded pieces (%#)', (text) => {
    const pieces = splitQwenInput(text);
    expect(pieces.join('')).toBe(text);
    expect(pieces.length).toBeGreaterThan(1);
    for (const piece of pieces) {
      expect(Array.from(piece).length).toBeLessThanOrEqual(600);
      expect(piece.length).toBeGreaterThan(0);
      expect(piece).not.toMatch(/[\uD800-\uDFFF]/u);
    }
  });
  it('prefers paragraph, then sentence, then whitespace boundaries in the latter half', () => {
    const start = 'a'.repeat(350);
    expect(splitQwenInput(start + '\n\n' + 'A sentence. ' + 'b'.repeat(700))[0]).toBe(
      start + '\n\n',
    );
    expect(splitQwenInput(start + '. ' + 'word '.repeat(100))[0]).toBe(start + '. ');
    expect(splitQwenInput('word '.repeat(150))[0]).toBe('word '.repeat(120));
  });
  it('never divides a word when a whitespace boundary is available', () => {
    const pieces = splitQwenInput('длинноеСлово '.repeat(100));
    expect(pieces.slice(0, -1).every((piece) => /\s$/u.test(piece))).toBe(true);
  });
  it('keeps oversized blank runs with a spoken neighbor instead of creating rejected blank requests', () => {
    for (const text of [
      ' '.repeat(1200) + 'A sentence. '.repeat(100),
      'A sentence. '.repeat(100) + ' '.repeat(1200),
      'A sentence. '.repeat(60) + '\n'.repeat(1200) + 'Another sentence. '.repeat(60),
    ]) {
      const pieces = splitQwenInput(text);
      expect(pieces.join('')).toBe(text);
      expect(pieces.every((piece) => piece.trim().length > 0)).toBe(true);
      expect(pieces.every((piece) => Array.from(piece.trim()).length <= 600)).toBe(true);
    }
    const blank = ' \t\n'.repeat(500);
    expect(splitQwenInput(blank)).toEqual([blank]);
  });
});
