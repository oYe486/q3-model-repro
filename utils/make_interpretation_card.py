"""Render sample 10's compact, paper-ready multimodal explanation figure."""

import base64
import json
import struct
import wave
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "attachment4"
ASSETS = OUT / "card_10_assets"


def waveform_svg(path, x, y, width, height, count=390):
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    duration = len(values) / rate
    span = max(1, len(values) // count)
    columns = []
    for index in range(count):
        segment = values[index * span:(index + 1) * span]
        peak = max((abs(value) for value in segment), default=0) / 32768
        amplitude = max(1.5, min(height / 2 - 5, peak * height * 0.8))
        xpos = x + index * width / count
        columns.append(
            f'<line x1="{xpos:.1f}" y1="{y + height / 2 - amplitude:.1f}" '
            f'x2="{xpos:.1f}" y2="{y + height / 2 + amplitude:.1f}" '
            f'stroke="#E49A40" stroke-width="2"/>'
        )
    return "".join(columns), duration


def main():
    records = [json.loads(line) for line in (OUT / "predictions_20.jsonl").read_text(encoding="utf-8").splitlines()]
    record = next(item for item in records if item["sample_id"] == "10")
    alignment = json.loads((OUT / "asr" / "10_base.en.json").read_text(encoding="utf-8"))["alignment"]
    evidence = record["primary_evidence"]
    loc = evidence["location"]
    score = record["predicted_score"]
    effects = {name: -value for name, value in record["modality_signed_support"].items()}

    assert record["predicted_class"] == 0 and record["main_modality"] == "text"
    assert loc["mapping_status"] == "text_exact"
    assert record["raw_text"][110:129] == "was pretty terrible" == loc["raw_text_fragment"]
    assert abs(sum(effects.values()) - score) < 1e-5
    indices = (22, 23, 24)
    words = alignment["raw_words"]
    asr = alignment["asr_words"]
    mapped = [asr[alignment["raw_to_asr"][str(i)]] for i in indices]
    assert [words[i]["normalized"] for i in indices] == [w["normalized"] for w in mapped]
    start, end = mapped[0]["start_seconds"], mapped[-1]["end_seconds"]
    assert (start, end) == (7.34, 9.02)

    audio_svg, duration = waveform_svg(ASSETS / "audio.wav", 80, 430, 870, 118)
    assert abs(duration - record["asr_audit"]["audio_duration_seconds"]) < 0.02
    frame = base64.b64encode((ASSETS / "frame_8.1s.jpg").read_bytes()).decode("ascii")
    assert start <= 8.1 <= end

    hx = 80 + 870 * start / duration
    hw = 870 * (end - start) / duration
    label = {"text": "文本", "audio": "语音", "vision": "视觉"}
    color = {"text": "#2878B8", "audio": "#E49A40", "vision": "#55A392"}
    rows = []
    zero_x, scale = 1010, 220
    for name, y in (("text", 677), ("audio", 731), ("vision", 785)):
        value = effects[name]
        x = zero_x + min(value, 0) * scale
        rows.append(f'<text x="80" y="{y + 7}" class="row">{label[name]}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y - 14}" width="{abs(value) * scale:.1f}" height="28" rx="4" fill="{color[name]}"/>')
        rows.append(f'<text x="1290" y="{y + 8}" class="number" text-anchor="end">{value:+.3f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="895" viewBox="0 0 1500 895">
<style>
  text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill:#243044; }}
  .meta {{ font-size:19px; fill:#657184; }}
  .result {{ font-size:27px; font-weight:700; }}
  .label {{ font-size:21px; font-weight:700; }}
  .transcript {{ font: 24px Arial, sans-serif; fill:#243044; }}
  .note {{ font-size:17px; fill:#657184; }}
  .row {{ font-size:21px; }}
  .number {{ font:700 22px Arial, sans-serif; }}
</style>
<rect width="1500" height="895" fill="#FFFFFF"/>
<text x="80" y="67" class="meta">附件4 · 对齐版 · 样本 10</text>
<text x="565" y="68" class="result">负向</text>
<text x="690" y="68" class="result">{score:.3f}</text>
<text x="1010" y="67" class="meta">主要参考模态</text>
<text x="1190" y="68" class="result" fill="#2878B8">文本</text>
<line x1="80" y1="93" x2="1420" y2="93" stroke="#D6DDE4"/>

<text x="80" y="145" class="label" fill="#2878B8">文本</text>
<text x="80" y="191" class="transcript">(umm) And you know I really do like to see</text>
<text x="80" y="232" class="transcript">fluffy chick flicks sometimes so I'm not</text>
<text x="80" y="273" class="transcript">against that but this one</text>
<rect x="80" y="288" width="273" height="43" rx="5" fill="#DDECF8"/>
<text x="92" y="319" class="transcript" font-weight="700" fill="#125E98">{escape(loc['raw_text_fragment'])}</text>
<text x="385" y="319" class="note">特征 [28, 31) · 原文 [110, 129)</text>

<text x="80" y="393" class="label" fill="#D3892E">原视频音轨</text>
<rect x="{hx:.1f}" y="422" width="{hw:.1f}" height="135" rx="4" fill="#DDECF8"/>
<line x1="80" y1="489" x2="950" y2="489" stroke="#E7EBEF"/>
{audio_svg}
<text x="80" y="580" class="note">0 s</text>
<text x="950" y="580" class="note" text-anchor="end">{duration:.2f} s</text>
<text x="{hx + hw / 2:.1f}" y="580" class="note" text-anchor="middle" fill="#2878B8">对应语音 7.34–9.02 s*</text>

<text x="1030" y="145" class="label" fill="#45917F">视频关键帧</text>
<rect x="1030" y="171" width="390" height="294" fill="#E9EEF1"/>
<image x="1030" y="171" width="390" height="294" href="data:image/jpeg;base64,{frame}"/>
<text x="1030" y="500" class="note">原视频第 243 帧 · 8.10 s*</text>
<text x="1030" y="535" class="note">* 按音轨逐词匹配选取的回看位置</text>

<line x1="80" y1="608" x2="1420" y2="608" stroke="#D6DDE4"/>
<text x="80" y="646" class="label">三模态对预测强度的贡献</text>
<line x1="1010" y1="657" x2="1010" y2="806" stroke="#9EABB7"/>
{''.join(rows)}
<text x="1010" y="830" class="note" text-anchor="middle">0</text>
<text x="80" y="856" class="note">遮挡高亮文本：{score:.3f} → {evidence['occluded_score']:.3f}（中性）</text>
<text x="1420" y="856" class="note" text-anchor="end">附件4无真实标签；音视频回看位置不是特征来源证明</text>
</svg>'''
    output = OUT / "interpretation_card_10.svg"
    output.write_text(svg, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
