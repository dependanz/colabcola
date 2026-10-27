"""Read a small real training subset from the official public subject ZIP using ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile

URL = "https://syncandshare.lrz.de/dl/fiRXRYvdGQoC162RZDDaZc/GaussianAvatars/release/218.zip"

class RangeFile(io.RawIOBase):
    def __init__(self, url, maximum=128*1024*1024):
        self.url, self.pos, self.transferred, self.maximum = url, 0, 0, maximum
        req=urllib.request.Request(url,headers={"Range":"bytes=0-0","Accept-Encoding":"identity"})
        with urllib.request.urlopen(req,timeout=30) as r:
            match=re.fullmatch(r"bytes 0-0/(\d+)",r.headers.get("Content-Range",""))
            if r.status!=206 or not match: raise ValueError("server must support exact byte ranges")
            self.size=int(match[1]); self.etag=r.headers.get("ETag"); r.read(1)
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if not 0<=self.pos<=self.size: raise ValueError("invalid seek")
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n==0:return b""
        if self.transferred+n>self.maximum: raise ValueError("download budget exceeded")
        start=self.pos; end=start+n-1
        headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
        if self.etag: headers["If-Match"]=self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=30) as r:
            if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{self.size}":
                raise ValueError("range response mismatch")
            if self.etag and r.headers.get("ETag")!=self.etag:raise ValueError("archive changed")
            body=r.read(n+1)
        if len(body)!=n:raise ValueError("truncated/oversized range")
        self.pos+=n;self.transferred+=n
        return body

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--list",action="store_true");args=p.parse_args()
    remote=RangeFile(URL)
    with zipfile.ZipFile(remote) as z:
        names=z.namelist();transforms=[n for n in names if n.endswith("transforms_train.json")]
        if args.list:
            print(json.dumps({"transforms":transforms,"members":len(names),"downloaded":remote.transferred},indent=2));return
        if args.output.exists():raise ValueError("output already exists")
        # First official training sequence; never choose a model checkpoint or test-only split.
        name=sorted(transforms)[0];prefix=name.rsplit("/",1)[0]
        data=json.loads(z.read(name));frames=data["frames"][:8]
        if len(frames)!=8:raise ValueError("need eight training views")
        canonical=z.getinfo(prefix+"/canonical_flame_param.npz")
        if not 0 < canonical.file_size <= 32*1024*1024:raise ValueError("invalid canonical FLAME member size")
        files={"canonical_flame_param.npz":z.read(canonical)};times={t:i for i,t in enumerate(sorted({f["timestep_index"] for f in frames}))}
        for f in frames:
            for field in ("file_path","flame_param_path"):
                rel=f[field]
                if field=="file_path" and ".png" not in rel:rel+=".png"
                path=PurePosixPath(rel)
                if path.is_absolute() or ".." in path.parts or "\\" in rel or ":" in rel:raise ValueError("unsafe member")
                info=z.getinfo(prefix+"/"+rel)
                if info.file_size>32*1024*1024:raise ValueError("oversized subset member")
                files[rel]=z.read(info) # zipfile verifies CRC32; hash each retained file below.
            f["timestep_index"]=times[f["timestep_index"]]
        centers={tuple(f["transform_matrix"][i][3] for i in range(3)) for f in frames}
        if len(centers)<2:raise ValueError("degenerate camera centers")
        args.output.mkdir(parents=True)
        records={}
        for rel,body in files.items():
            dest=args.output/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
            records[rel]={"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}
        for split in ("train","val","test"):
            (args.output/f"transforms_{split}.json").write_text(json.dumps(dict(data,frames=frames if split=="train" else []),indent=2),encoding="utf-8")
        report={"url":URL,"archive_bytes":remote.size,"etag":remote.etag,"transferred_bytes":remote.transferred,
                "source_transform":name,"train_frames":8,"timesteps":len(times),"files":records,
                "integrity":"HTTP range identity, per-member ZIP CRC32 and retained SHA256; full archive hash not downloaded"}
        (args.output/"acquisition.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))

if __name__=="__main__":main()
