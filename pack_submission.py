import os, zipfile, argparse
def zipdir(path, ziph, rel=""):
    for root,_,files in os.walk(path):
        for f in files:
            p=os.path.join(root,f)
            arc=os.path.join(rel, os.path.relpath(p,path))
            ziph.write(p, arcname=arc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--masks", default="masks")
    ap.add_argument("--solution", default="solution")
    args=ap.parse_args()

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        zipdir(args.masks, z, rel="masks")
        zipdir(args.solution, z, rel="solution")
    print("wrote", args.out)

if __name__=="__main__":
    main()