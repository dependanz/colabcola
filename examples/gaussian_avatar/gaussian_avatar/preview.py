"""Compose honest input/reconstruction previews from upstream renderer outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(source, output, views):
    from PIL import Image, ImageDraw, ImageFont
    if output.exists():
        raise ValueError("preview output already exists")
    names = [f"{i:05d}.png" for i in range(views)]
    for folder in ("gt", "renders"):
        if sorted(p.name for p in (source/folder).glob("*.png")) != names:
            raise ValueError("missing or unexpected rendered views: " + folder)
    images = []
    records = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default(size=16)
    size = None
    for i, name in enumerate(names):
        with Image.open(source/'gt'/name) as f: gt = f.convert('RGB')
        with Image.open(source/'renders'/name) as f: pred = f.convert('RGB')
        if gt.size != pred.size or (size and size != gt.size):
            raise ValueError("input/reconstruction dimensions differ")
        size = gt.size
        width, height = size
        pair = Image.new('RGB', (width*2, height+36), 'white')
        draw = ImageDraw.Draw(pair)
        draw.text((6,8), f"View {i+1} / Input", fill='black', font=font)
        draw.text((width+6,8), "Reconstruction / step 200", fill='black', font=font)
        pair.paste(gt,(0,36)); pair.paste(pred,(width,36))
        images.append(pair)
        records.append({'view':i, 'input_sha256':sha(source/'gt'/name),
                        'render_sha256':sha(source/'renders'/name), 'width':width,'height':height})
    output.mkdir(parents=True)
    for i, pair in enumerate(images): pair.save(output/f'pair-{i:02d}.png')
    width,height=images[0].size
    overview=Image.new('RGB',(width*2,height*((views+1)//2)), '#eeeeee')
    for i,pair in enumerate(images):overview.paste(pair,((i%2)*width,(i//2)*height))
    overview.save(output/'overview.png')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/preview'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--views',type=int,default=8)
    args=p.parse_args()
    if not 1<=args.views<=16:p.error('views must be1..16')
    hashes=json.loads(args.expected.read_text(encoding='utf-8'))
    for name,expected in hashes.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(args.model/relative)!=expected:
            raise ValueError('saved model differs: '+name)
    records=compose(args.source,args.output,args.views)
    report={'state':'rendered','iteration':200,'training_performed':False,
            'model_files_unchanged':True,'model_sha256':hashes,'views':records,
            'comparison':'Training views at training resolution; input left, reconstruction right. No retraining or held-out quality claim.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
