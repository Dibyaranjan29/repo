import asyncio
import io
import re
import shutil
import subprocess
import wave
from array import array

import streamlit as st

try:
    import edge_tts
except ImportError:
    edge_tts = None

st.set_page_config(
    page_title="Devil's Dictation",
    page_icon="📝",
    layout="wide",
)

STRIP_PATTERN = re.compile(r"[,.:;!?\"“”()\[\]{}]")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "shri", "smt", "st", "jr", "sr", "hon", "gen", "col", "capt", "sri"}


def count_words(text):
    return len(text.split())


MAX_CHUNK_WORDS = 10
SPOKEN_MARKS = {
    ",": "comma",
    ";": "semicolon",
    ":": "colon",
    ".": "fullstop",
    "!": "exclamation mark",
    "?": "question mark",
}
TRAILING_MARK = re.compile(r"([,;:.!?])([\"'”’)\]}]*)$")


def classify_token(token):
    core = token.rstrip("\"'”’)]}")
    if not core:
        return None
    if core.endswith(("!", "?")):
        return "sentence"
    if core.endswith("."):
        bare = core[:-1]
        if bare.lower() in ABBREVIATIONS:
            return None
        if re.fullmatch(r"[A-HJ-Z]", bare):
            return None
        if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", bare):
            return None
        return "sentence"
    if core.endswith((",", ";", ":")):
        return "clause"
    return None


def speak_marks_aloud(token):
    match = TRAILING_MARK.search(token)
    if not match:
        return token
    mark = match.group(1)
    return token[: match.start()] + " " + SPOKEN_MARKS[mark]


def build_chunks(text, speak_punctuation):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    current = []

    def flush(end_type):
        if not current:
            return
        words = list(current)
        spoken = list(words)
        if speak_punctuation:
            spoken[-1] = speak_marks_aloud(spoken[-1])
        chunks.append({"text": " ".join(spoken), "words": len(words), "end": end_type})
        current.clear()

    for token in cleaned.split(" "):
        current.append(token)
        end_type = classify_token(token)
        if end_type is None and len(current) >= MAX_CHUNK_WORDS:
            end_type = "none"
        if end_type is not None:
            flush(end_type)
    flush("sentence")
    return chunks


def count_sentences(chunks):
    return sum(1 for chunk in chunks if chunk["end"] == "sentence")


def tokenize(text):
    words = []
    starts = []
    at_sentence_start = True
    for raw in text.split():
        cleaned = STRIP_PATTERN.sub("", raw)
        if cleaned:
            words.append(cleaned)
            starts.append(at_sentence_start)
            at_sentence_start = False
        if raw[-1] in ".!?" and cleaned.lower() not in ABBREVIATIONS:
            at_sentence_start = True
    return words, starts


def levenshtein_alignment(master, typed):
    m = len(master)
    n = len(typed)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if master[i - 1].lower() == typed[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    aligned = []
    omissions = []
    insertions = []
    substitutions = []
    i = m
    j = n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if master[i - 1].lower() == typed[j - 1].lower() else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    aligned.append((i - 1, j - 1))
                else:
                    substitutions.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            omissions.append(i - 1)
            i -= 1
            continue
        insertions.append(j - 1)
        j -= 1

    aligned.reverse()
    omissions.reverse()
    insertions.reverse()
    substitutions.reverse()
    return dp[m][n], aligned, omissions, insertions, substitutions


def is_pronoun_i(word):
    return word == "I" or (word.startswith("I'") and len(word) > 2)


def find_half_mistakes(master, typed, starts, aligned):
    pronoun_lapses = []
    proper_noun_lapses = []
    for master_idx, typed_idx in aligned:
        master_word = master[master_idx]
        typed_word = typed[typed_idx]
        if master_word == typed_word:
            continue
        if is_pronoun_i(master_word):
            if typed_word[0] == "i":
                pronoun_lapses.append((master_word, typed_word))
            continue
        if (
            master_word[0].isupper()
            and not starts[master_idx]
            and typed_word == typed_word.lower()
        ):
            proper_noun_lapses.append((master_word, typed_word))
    return pronoun_lapses, proper_noun_lapses


SAMPLE_RATE = 24000
MAX_SPEED = 2.0
VOICES = {
    "Neerja (Indian English, Female)": "en-IN-NeerjaNeural",
    "Prabhat (Indian English, Male)": "en-IN-PrabhatNeural",
    "Ryan (British English, Male)": "en-GB-RyanNeural",
    "Sonia (British English, Female)": "en-GB-SoniaNeural",
    "Guy (US English, Male)": "en-US-GuyNeural",
    "Jenny (US English, Female)": "en-US-JennyNeural",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def pcm_ms(pcm):
    return len(pcm) / (2 * SAMPLE_RATE) * 1000.0


def silence_pcm(ms):
    return b"\x00\x00" * int(SAMPLE_RATE * ms / 1000.0)


def to_samples(pcm):
    samples = array("h")
    samples.frombytes(bytes(pcm[: len(pcm) - (len(pcm) % 2)]))
    return samples


def trim_pcm(pcm, threshold=60, margin_ms=40):
    samples = to_samples(pcm)
    total = len(samples)
    start = 0
    while start < total and abs(samples[start]) < threshold:
        start += 1
    if start == total:
        return b""
    end = total - 1
    while end > start and abs(samples[end]) < threshold:
        end -= 1
    margin = int(SAMPLE_RATE * margin_ms / 1000)
    start = max(0, start - margin)
    end = min(total, end + 1 + margin)
    return samples[start:end].tobytes()


def get_ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def decode_mp3_to_pcm(data, ffmpeg):
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=True)
    return result.stdout


def encode_audio(pcm, ffmpeg):
    if ffmpeg:
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-f",
            "mp3",
            "pipe:1",
        ]
        result = subprocess.run(command, input=bytes(pcm), capture_output=True)
        if result.returncode == 0 and result.stdout:
            return result.stdout, "audio/mpeg"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(pcm))
    return buffer.getvalue(), "audio/wav"


async def fetch_tts_mp3(text, voice, multiplier):
    percent = int(round((clamp(multiplier, 0.5, MAX_SPEED) - 1.0) * 100))
    rate = f"{percent:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    data = bytearray()
    async for message in communicate.stream():
        if message["type"] == "audio":
            data.extend(message["data"])
    if not data:
        raise RuntimeError("The speech service returned no audio.")
    return bytes(data)


def speed_key(speed):
    return round(clamp(speed, 0.5, MAX_SPEED) * 20) / 20


async def fetch_word_audio(token, voice, speed, ffmpeg, cache, gate):
    if not re.search(r"\w", token):
        return b""
    key = (token, voice, speed_key(speed))
    if key in cache:
        return cache[key]
    pcm = b""
    for attempt in range(3):
        try:
            async with gate:
                data = await fetch_tts_mp3(token, voice, key[2])
            pcm = trim_pcm(decode_mp3_to_pcm(data, ffmpeg))
            break
        except Exception:
            await asyncio.sleep(0.4 * (attempt + 1))
    cache[key] = pcm
    return pcm


async def fetch_many(tokens, voice, speed, ffmpeg, cache, gate, on_step=None):
    unique = list(dict.fromkeys(tokens))

    async def one(token):
        pcm = await fetch_word_audio(token, voice, speed, ffmpeg, cache, gate)
        if on_step:
            on_step()
        return pcm

    await asyncio.gather(*(one(token) for token in unique))


def sentence_groups(chunks):
    groups = []
    start = 0
    for index, chunk in enumerate(chunks):
        if chunk["end"] == "sentence" or index == len(chunks) - 1:
            groups.append(list(range(start, index + 1)))
            start = index + 1
    return groups


def pause_plan(chunks, group, budget_ms, pause_ms):
    raw = []
    for k in group:
        if chunks[k]["end"] == "sentence":
            raw.append(pause_ms)
        elif chunks[k]["end"] == "clause":
            raw.append(pause_ms * 0.5)
        else:
            raw.append(0.0)
    total = sum(raw)
    scale = 1.0
    if total > 0:
        scale = min(1.0, budget_ms * 0.6 / total)
    return [value * scale for value in raw]


def join_words(words, gap_ms):
    if not words:
        return b""
    gap = silence_pcm(gap_ms)
    return gap.join(words)


async def render_dictation_audio(chunks, wpm, voice, lead_in_seconds, sentence_pause_ms, base_speed, ffmpeg, on_progress):
    lead_ms = lead_in_seconds * 1000.0
    pcm = bytearray(silence_pcm(lead_ms))
    cumulative_budget = 0.0
    report = []
    done = 0
    cache = {}
    gate = asyncio.Semaphore(8)

    all_tokens = [token for chunk in chunks for token in chunk["text"].split()]
    unique_total = max(1, len(set(all_tokens)))
    fetched = [0]

    def on_step():
        fetched[0] += 1
        on_progress(min(0.85, 0.85 * fetched[0] / unique_total))

    await fetch_many(all_tokens, voice, base_speed, ffmpeg, cache, gate, on_step)

    def lookup(token, speed):
        return cache.get((token, voice, speed_key(speed)), b"")

    for group in sentence_groups(chunks):
        group_words = sum(chunks[k]["words"] for k in group)
        budget_ms = group_words / wpm * 60000.0
        pauses = pause_plan(chunks, group, budget_ms, sentence_pause_ms)
        lag_ms = pcm_ms(pcm) - lead_ms - cumulative_budget
        available_ms = max(budget_ms * 0.35, budget_ms - sum(pauses) - lag_ms)
        group_tokens = [token for k in group for token in chunks[k]["text"].split()]

        speed = speed_key(base_speed)
        natural_ms = sum(pcm_ms(lookup(token, speed)) for token in group_tokens)
        for attempt in range(3):
            if natural_ms <= available_ms or speed >= MAX_SPEED:
                break
            speed = speed_key(clamp(speed * natural_ms / available_ms + 0.05, speed + 0.05, MAX_SPEED))
            await fetch_many(group_tokens, voice, speed, ffmpeg, cache, gate)
            natural_ms = sum(pcm_ms(lookup(token, speed)) for token in group_tokens)

        boundaries = sum(max(0, len([t for t in chunks[k]["text"].split() if lookup(t, speed)]) - 1) for k in group)
        gap_ms = max(0.0, available_ms - natural_ms) / boundaries if boundaries else 0.0

        for position, k in enumerate(group):
            words = [lookup(token, speed) for token in chunks[k]["text"].split()]
            words = [word for word in words if word]
            pcm.extend(join_words(words, gap_ms))
            pcm.extend(silence_pcm(pauses[position]))
            done += 1
            report.append(
                {
                    "Segment": done,
                    "Text": chunks[k]["text"],
                    "Words": chunks[k]["words"],
                    "Voice Speed (x)": round(speed, 2),
                    "Word Gap (ms)": int(round(gap_ms)),
                    "Pause After (ms)": int(round(pauses[position])),
                }
            )
        cumulative_budget += budget_ms
        on_progress(0.85 + 0.15 * done / len(chunks))

    audio_ms = pcm_ms(pcm) - lead_ms
    audio_bytes, mime = encode_audio(pcm, ffmpeg)
    return {
        "audio": audio_bytes,
        "mime": mime,
        "report": report,
        "audio_ms": audio_ms,
    }


def generate_audio_blocking(chunks, wpm, voice, lead_in_seconds, sentence_pause_ms, base_speed, on_progress):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found. Run: pip install imageio-ffmpeg")
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            render_dictation_audio(chunks, wpm, voice, lead_in_seconds, sentence_pause_ms, base_speed, ffmpeg, on_progress)
        )
    finally:
        loop.close()


def render_dictation_mode():
    st.title("🎙️ Dictation Engine")
    st.caption("Server-rendered dictation audio with exact timing. No browser timers, no speech lag.")

    if edge_tts is None:
        st.error("Missing package. Install with: pip install edge-tts imageio-ffmpeg")
        return

    passage = st.text_area(
        "Practice Passage / Shorthand Outline Source",
        height=260,
        placeholder="Paste your custom practice passage here...",
        key="dictation_passage",
    )

    wpm = st.slider(
        "Dictation Speed (WPM)",
        min_value=40,
        max_value=130,
        value=80,
        step=1,
        help="Grade D baseline is 80 WPM.",
    )

    option_a, option_b = st.columns(2)
    with option_a:
        voice_label = st.selectbox("Voice", list(VOICES.keys()))
    with option_b:
        lead_in = st.slider("Lead-in silence (seconds)", min_value=0, max_value=10, value=3)

    pause_seconds = st.slider(
        "Pause after each sentence (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Fixed at every WPM. Commas, semicolons and colons get half of this.",
    )

    voice_speed = st.slider(
        "Voice reading speed (x)",
        min_value=0.7,
        max_value=1.3,
        value=1.0,
        step=0.05,
        help="Natural speed of each word. The voice only speeds up above this when the WPM target needs it. Lower WPM adds gaps between words instead of slowing the voice.",
    )

    speak_punctuation = st.checkbox(
        "Read punctuation marks aloud (comma, full stop, question mark...)",
        value=False,
        help="Off: marks are not spoken but create natural pauses and intonation. On: marks are spoken as words.",
    )

    total_words = count_words(passage)
    duration_minutes = total_words / wpm if wpm else 0.0
    chunks = build_chunks(passage, speak_punctuation)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Words", total_words)
    col_b.metric("Target Speed", f"{wpm} WPM")
    col_c.metric("Estimated Duration", f"{duration_minutes:.2f} min")
    col_d.metric("Sentences / Segments", f"{count_sentences(chunks)} / {len(chunks)}")

    settings_key = (passage, wpm, voice_label, lead_in, speak_punctuation, pause_seconds, voice_speed)

    if st.button("Generate Dictation Audio", type="primary", disabled=not chunks):
        progress_bar = st.progress(0.0, text="Rendering dictation audio...")

        def on_progress(fraction):
            progress_bar.progress(min(1.0, fraction), text=f"Rendering dictation audio... {int(fraction * 100)}%")

        try:
            result = generate_audio_blocking(chunks, wpm, VOICES[voice_label], lead_in, pause_seconds * 1000.0, voice_speed, on_progress)
        except Exception as error:
            progress_bar.empty()
            st.error(f"Audio generation failed: {error}")
            return
        progress_bar.empty()
        result["key"] = settings_key
        result["words"] = total_words
        st.session_state["dictation_result"] = result

    result = st.session_state.get("dictation_result")
    if not result:
        st.info("Paste a passage, choose the speed and voice, then generate the audio.")
        return

    if result["key"] != settings_key:
        st.warning("Inputs changed since this audio was generated. Click Generate to refresh.")

    audio_minutes = result["audio_ms"] / 60000.0
    effective_wpm = result["words"] / audio_minutes if audio_minutes > 0 else 0.0

    r1, r2, r3 = st.columns(3)
    r1.metric("Audio Length (after lead-in)", f"{audio_minutes:.2f} min")
    r2.metric("Effective Pace", f"{effective_wpm:.1f} WPM")
    r3.metric("Lead-in", f"{result['key'][3]} s")

    st.audio(result["audio"], format=result["mime"])
    extension = "mp3" if result["mime"] == "audio/mpeg" else "wav"
    st.download_button(
        "Download Dictation Audio",
        data=result["audio"],
        file_name=f"dictation_{result['key'][1]}wpm.{extension}",
        mime=result["mime"],
    )

    with st.expander("Segment Timing Report"):
        st.dataframe(result["report"], use_container_width=True)


def render_evaluator_mode():
    st.title("📊 SSC Error Evaluator")
    st.caption("Compare the master passage with your transcription using the SSC marking scheme.")

    left, right = st.columns(2)
    with left:
        master_text = st.text_area(
            "Master Reference Text (Original Passage)",
            height=320,
            key="master_text",
        )
    with right:
        typed_text = st.text_area(
            "Your Typed Shorthand Transcription Script",
            height=320,
            key="typed_text",
        )

    if not st.button("Run Evaluation Report Audit", type="primary"):
        return

    master_words, master_starts = tokenize(master_text)
    typed_words, _ = tokenize(typed_text)

    if not master_words:
        st.error("Please provide the Master Reference Text.")
        return

    full_total, aligned, omissions, insertions, substitutions = levenshtein_alignment(
        master_words, typed_words
    )
    pronoun_lapses, proper_noun_lapses = find_half_mistakes(
        master_words, typed_words, master_starts, aligned
    )

    full_mistakes = full_total
    half_mistakes = len(pronoun_lapses) + len(proper_noun_lapses)
    total_master = len(master_words)
    effective_errors = full_mistakes + (half_mistakes / 2)
    error_percentage = (effective_errors / total_master) * 100

    st.subheader("Scorecard Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Master Words", total_master)
    c2.metric("Total Effective Errors", f"{effective_errors:.1f}")

    if error_percentage <= 5.0:
        status_label = "Excellent (Passed Grade C)"
        delta_color = "normal"
    elif error_percentage <= 7.0:
        status_label = "Safe for Grade D"
        delta_color = "off"
    else:
        status_label = "Exceeds Exam Limit Boundaries"
        delta_color = "inverse"

    c3.metric(
        "Official SSC Error Rate",
        f"{error_percentage:.2f}%",
        delta=status_label,
        delta_color=delta_color,
    )

    if error_percentage <= 5.0:
        st.success(status_label)
    elif error_percentage <= 7.0:
        st.warning(status_label)
    else:
        st.error(status_label)

    with st.expander("Detailed Mistake Breakdown", expanded=True):
        st.markdown("**Full Mistakes (1.0 each)**")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Omissions", len(omissions))
        f2.metric("Insertions", len(insertions))
        f3.metric("Substitutions", len(substitutions))
        f4.metric("Total Full", full_mistakes)

        st.markdown("**Half Mistakes (0.5 each)**")
        h1, h2, h3 = st.columns(3)
        h1.metric("Pronoun 'I' Lapses", len(pronoun_lapses))
        h2.metric("Proper Noun Case Errors", len(proper_noun_lapses))
        h3.metric("Total Half", half_mistakes)

        rows = []
        for idx in omissions:
            rows.append({"Type": "Omission", "Master Word": master_words[idx], "Typed Word": "-"})
        for idx in insertions:
            rows.append({"Type": "Insertion", "Master Word": "-", "Typed Word": typed_words[idx]})
        for master_idx, typed_idx in substitutions:
            rows.append(
                {
                    "Type": "Substitution",
                    "Master Word": master_words[master_idx],
                    "Typed Word": typed_words[typed_idx],
                }
            )
        for master_word, typed_word in pronoun_lapses:
            rows.append({"Type": "Half: Pronoun I", "Master Word": master_word, "Typed Word": typed_word})
        for master_word, typed_word in proper_noun_lapses:
            rows.append({"Type": "Half: Proper Noun", "Master Word": master_word, "Typed Word": typed_word})

        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.success("No mistakes detected. Perfect transcription!")

        st.caption(
            f"Formula: (({full_mistakes} + ({half_mistakes} / 2)) / {total_master}) × 100 = {error_percentage:.2f}%"
        )


def main():
    st.sidebar.title("SSC Steno Suite")
    mode = st.sidebar.radio(
        "Select Operating Mode",
        ["Dictation Engine Mode", "SSC Error Evaluator Mode"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Grade C: up to 5% errors | Grade D: up to 7% errors")

    if mode == "Dictation Engine Mode":
        render_dictation_mode()
    else:
        render_evaluator_mode()


main()
