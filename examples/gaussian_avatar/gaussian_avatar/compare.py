"""Compare two saved training-view render sets; no model execution or packaging."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(model, output, before=200, after=5000, views=8):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    if not 0 < before < after or not 1 <= views <= 16:
        raise ValueError("invalid comparison iterations or views")
    if output.exists():
        raise ValueError("comparison destination already exists")
    names = [f"{i:05d}.png" for i in range(views)]
    roots = [model / "train" / f"ours_{step}" for step in (before, after)]
    for root in roots:
        for folder in ("gt", "renders"):
            if sorted(p.name for p in (root/folder).glob("*.png")) != names:
                raise ValueError("missing or unexpected rendered views")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default(size=16)
    records, panels = [], []
    for index, name in enumerate(names):
        images = []
        for path in (roots[0]/"gt"/name, roots[1]/"gt"/name,
                     roots[0]/"renders"/name, roots[1]/"renders"/name):
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        arrays = [np.asarray(im).astype(np.float64)/255 for im in images]
        if any(a.shape != arrays[0].shape for a in arrays) or not np.array_equal(arrays[0], arrays[1]):
            raise ValueError("comparison inputs or dimensions differ")
        mask = np.min(arrays[0], axis=2) < .98
        row = {"view": index, "input_sha256": hashlib.sha256((roots[0]/"gt"/name).read_bytes()).hexdigest(),
               "foreground_pixels": int(mask.sum()), "width": images[0].width, "height": images[0].height}
        for step, array in zip((before, after), arrays[2:]):
            error = (array-arrays[0])**2
            mse = float(error.mean())
            row[str(step)] = {"rgb_mse": mse, "psnr_db": -10*math.log10(mse) if mse else None,
                              "foreground_mse": float(error[mask].mean()) if mask.any() else None}
        width, height = images[0].size
        panel = Image.new("RGB", (3*width, height+36), "white")
        draw = ImageDraw.Draw(panel)
        for col, (im, label) in enumerate(zip((images[0],images[2],images[3]),
                (f"View {index+1} / Input",f"Step {before}",f"Step {after}"))):
            draw.text((col*width+6,8),label,fill="black",font=font)
            panel.paste(im,(col*width,36))
        panels.append(panel);records.append(row)
    output.mkdir(parents=True)
    for index,panel in enumerate(panels):panel.save(output/f"comparison-{index:02d}.png")
    width,height=panels[0].size
    overview=Image.new("RGB",(width*2,height*((views+1)//2)),"#eeeeee")
    for index,panel in enumerate(panels):overview.paste(panel,((index%2)*width,(index//2)*height))
    overview.save(output/"overview.png")
    means={str(step):sum(row[str(step)]["rgb_mse"] for row in records)/views for step in (before,after)}
    report={"state":"verified", "before":before,"after":after,"views":records,
            "mean_rgb_mse":means,"mean_mse_reduction_fraction":1-means[str(after)]/means[str(before)] if means[str(before)] else None,
            "boundary":"Same eight training views and resolution, quantized RGB PNG metrics. No held-out or animation-quality claim. Foreground mask is input min(RGB)<0.98; perfect-match PSNR is null."}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,default=Path("outputs/avatar"))
    p.add_argument("--output",type=Path,default=Path("outputs/comparison"))
    p.add_argument("--before",type=int,default=200)
    p.add_argument("--after",type=int,default=5000)
    p.add_argument("--views",type=int,default=8)
    args=p.parse_args()
    print(json.dumps(compare(args.model,args.output,args.before,args.after,args.views),indent=2))

if __name__=="__main__":main()
