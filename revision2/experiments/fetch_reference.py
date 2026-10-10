"""Download unmodified MIT-licensed independent simulator at a pinned commit."""
from common import HERE, dump, sha
import requests

COMMIT = 'ad254f31500a71374eb28c85a0905d9fa2ae2c28'
BASE = f'https://raw.githubusercontent.com/neurospin/pymultifracs/{COMMIT}/'


def main():
    out = HERE / 'vendor' / 'pymultifracs'
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for src in ['LICENSE', 'pymultifracs/simul/pzutils.py',
                'pymultifracs/simul/fbm.py', 'pymultifracs/simul/mrw.py']:
        response = requests.get(BASE + src, timeout=60)
        response.raise_for_status()
        path = out / src.rsplit('/', 1)[-1]
        path.write_bytes(response.content)
        rows.append(dict(url=BASE + src, local_name=path.name, sha256=sha(path)))
    dump(out / 'provenance.json', dict(repository='https://github.com/neurospin/pymultifracs',
         commit=COMMIT, retrieved_on='2026-10-07', files=rows,
         modifications='None. Loaded as a namespace package without package installation.',
         selected_method='cme; independent scipy Cholesky reference implemented separately'))
    print('Downloaded pinned independent simulator:', out, flush=True)


if __name__ == '__main__':
    main()
