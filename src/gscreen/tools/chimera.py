import shutil
import subprocess as sp
from contextlib import AbstractContextManager
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Set

from .. import utils
from ..typing import StrPath

__all__ = ["Chimera"]


class _ChimeraScriptWrapper:
    EXEC_CHIMERA_NOGUI = ("/opt/chimera/current/bin/chimera", "--nogui")
    DEFAULT_IMPORTS = {
        "from Midas import write",
        "from chimera import openModels, runCommand as rc",
    }

    imports: Set[str]
    script: List[str]

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.imports = self.DEFAULT_IMPORTS.copy()
        self.script = []

    def rc(self, args: str):
        self.script.append(f"rc({args!r})")

    def del_matching(self, spec: str):
        self.rc(f"del {spec}")

    def merge(self, spec: str):
        self.rc(f"combine {spec}")

    def align(self, ref: int = 0, alg: str = "sw"):
        self.imports |= {
            "from MatchMaker import cmdMatch as match",
            "from chimera.selection import OSLSelection",
        }
        self.script.append(f"""\
ref = OSLSelection({f"#{ref}"!r})
match(ref, OSLSelection('#'), alg={str(alg)!r})""")

    def addh(self, hbond=False):
        prefix = "hbond" if hbond else "simple"
        self.imports.add(f"from AddH import {prefix}AddHydrogens as addh")
        self.script.append("""\
exclude = []
all_models = openModels.list()
for mol in all_models:
    try:
        addh([mol], inIsolation=True)
    except Exception as e:
        exclude.append(mol)
if exclude:
    print "Failed to add hydrogens to: ", len(exclude), "/", len(all_models)
    openModels.close(exclude)""")

    def addchg(self, method: str = "gas"):
        if not method.lower().startswith(("gas", "am1")):
            raise ValueError(f"Unknown charge method {method}")

        self.imports.add("from AddCharge import cmdAddAllCharge as addchg")
        self.script.append(f"""\
exclude = []
all_models = openModels.list()
for mol in all_models:
    try:
        addchg(molecules=[mol], method={method!r})
    except Exception as e:
        exclude.append(mol)
if exclude:
    print "Failed to add charges to: ", len(exclude), "/", len(all_models)
    openModels.close(exclude)""")

    def open(self, fpath: StrPath, fmt: str = None):
        self.script.append(f"openModels.open({str(fpath)!r}, type={fmt!r})")

    def write(
        self,
        fpath: StrPath,
        model_idx: List[int] = None,
        ref_idx: int = None,
        fmt: str = None,
    ):
        if model_idx is None:
            model_arg = "models"
        else:
            model_arg = f"[models[i] for i in {model_idx!r}]"

        if ref_idx is None:
            ref_arg = "None"
        else:
            ref_arg = f"models[{ref_idx}]"

        if fmt is None:
            if not isinstance(fpath, Path):
                fpath = Path(fpath)
            fmt = fpath.suffix.lstrip(".")

        self.script.append("models = openModels.list()")
        self.script.append(
            (
                f"write({model_arg}, {ref_arg}, {str(fpath)!r}, "
                f"format={fmt!r})"
            )
        )

    def run(self, verbose: bool = False, keep: bool = True):
        chimera = list(self.EXEC_CHIMERA_NOGUI)
        if verbose:
            stdout = None
        else:
            chimera.append("--silent")
            stdout = sp.DEVNULL
        stderr = stdout

        with NamedTemporaryFile("w", suffix=".py") as f:
            f.write("\n".join(chain(self.imports, self.script)))
            f.flush()
            sp.run(
                chimera + ["--script", f.name],
                check=True,
                stdout=stdout,
                stderr=stderr,
            )

        if not keep:
            self.reset()


class Chimera(AbstractContextManager):
    def __init__(self, verbose: bool = False, keep: bool = False):
        super().__init__()
        self.wrapper = _ChimeraScriptWrapper()
        self.verbose = verbose
        self.keep = keep

    def __enter__(self):
        super().__enter__()
        return self.wrapper

    def __exit__(self, *exc_args):
        if all(arg is None for arg in exc_args):
            self.wrapper.run(verbose=self.verbose, keep=self.keep)

        return super().__exit__(*exc_args)

    def addh(
        self,
        query: StrPath,
        result: StrPath,
        force: bool = True,
        delete: bool = True,
    ):
        with self as chimera:
            chimera.open(query)
            if delete:
                chimera.del_matching("H")
            chimera.addh()
            chimera.write(result)

    def addchg(
        self,
        query: StrPath,
        result: StrPath,
        force: bool = True,
        method: str = "gas",
        addh: bool = True,
        delete: bool = True,
    ):
        with self as chimera:
            chimera.open(query)
            if addh:
                if delete:
                    chimera.del_matching("H")
                chimera.addh()
            chimera.addchg(method=method)
            chimera.write(result)

    def convert(
        self,
        query: StrPath,
        result: StrPath,
        force: bool = True,
        ifmt: str = None,
        ofmt: str = None,
    ):
        query, result, force = utils.check_query_result(query, result, force)
        if not force and result.exists():
            raise FileExistsError(str(result))

        if not ifmt:
            ifmt = query.suffix[1:]
        if not ofmt:
            ofmt = result.suffix[1:]

        if ifmt == ofmt:
            shutil.copy(query, result)
            return

        with self as chimera:
            chimera.open(query, fmt=ifmt)
            chimera.write(result, fmt=ofmt)
