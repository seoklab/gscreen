# G-screen

A Scalable Receptor-Aware Virtual Screening through Flexible Ligand Alignment.

## Installation

G-screen runs in a virtual environment, and works with Python 3.10+. We
recommend using uv to manage the virtual environment. G-screen works with
G-align as an extra dependency that need to be installed separately; see below
for details.

### Python environment

This will create a virtual environment in `.venv` directory at the project root,
and install the required dependencies. G-screen will be installed during this
process (in editable mode).

```bash
uv sync [--no-dev]
```

### G-align

G-align could be downloaded in a precompiled binary form
[here](https://drive.google.com/file/d/12TAwz-y2EyE68MQD7_TV-nxm1mCHtW5l/view?usp=share_link),
or compiled from source code available [on GitHub](https://github.com/seoklab/galign).

G-align is Linux-only and requires `libomp5` to be installed. On Ubuntu, this
can be done with:

```bash
sudo apt-get install libomp5
```

G-align must be available in `$PATH` environment variable for G-screen to
work.

## Examples

### DUD-E SAHH target

See [DUD-E SAHH target](examples/dude-sahh/README.md).

## License and Disclaimer

G-screen is distributed under the GNU General Public License v2.0 (GPL-2.0-only).

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program in a file named `LICENSE`; if not, see
<https://www.gnu.org/licenses/>.
